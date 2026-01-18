from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.metrics import F1, Accuracy
from ontolearn.quality_funcs import evaluate_concept
from ontolearn.learning_problem import PosNegLPStandard
from owlapy import owl_expression_to_dl
from gymnasium import spaces
from scipy.special import softmax
import functools
from copy import copy

import random
import pandas as pd
from collections import defaultdict

from catboost import CatBoostClassifier, Pool
import numpy as np
import json
import tempfile
import os

from belief_functions.BFT import *
from encoders.expression_encoders import ExpressionEncoder
from utils.utils import *
from owlapy.class_expression import OWLDataHasValue

from owlapy.owl_literal import OWLLiteral
from owlapy.owl_property import OWLDataPropertyExpression
from pettingzoo import ParallelEnv


class Concept_Dec_POMPD(ParallelEnv):
    metadata = {
        "name": "concept_dec_pompd_v0", "render_modes": ["human"]
    }

    def __init__(
            self,
            file_path,
            pos,
            neg,
            reward_type,
            max_action,
            render=False,
            render_mode=None,
            max_card=10,
            device='cpu'
    ):
        self.cntr = 0
        self.device = device
        self.filepath = file_path
        self.kb = KnowledgeBase(path=self.filepath)
        self.render_mode = render_mode
        self.cg = self.kb.generator
        self.pos = pos
        self.neg = neg
        self.f1_score = 0

        lp = PosNegLPStandard(pos=pos, neg=neg)
        lp.all = pos.union(neg)
        self.lp = lp

        self.reward_type = reward_type
        self.max_action = max_action

        self.set_operators = [
            self.cg.union,
            self.cg.intersection,
            self.cg.negation,
        ]
        self.quantifiers = [
            self.cg.existential_restriction,
            self.cg.universal_restriction,
            self.cg.has_value_restriction
        ]
        self.cardinality = [
            self.cg.min_cardinality_restriction,
            self.cg.max_cardinality_restriction,
            self.cg.exact_cardinality_restriction
        ]
        self.numeric_dp = self._get_numeric_splits()

        self.concepts = list(self.kb.concepts)
        self.properties = list(self.kb.object_properties) + list(self.kb.data_properties)

        self.mappings = {
            "concepts": dict(
                [(idx, val) for idx, val in enumerate(self.concepts)]
            ),
            "properties": dict(
                [(idx, val) for idx, val in enumerate(self.properties)]
            ),
            "set": dict(
                [(idx, val) for idx, val in enumerate(self.set_operators)]
            ),
            "quantifier": dict(
                [(idx, val) for idx, val in enumerate(self.quantifiers)]
            ),
            "cardinality": dict(
                [(idx, val) for idx, val in enumerate(self.cardinality)]
            )
        }
        self.max_card = max_card
        self.possible_agents = ["a_0", "a_1"]
        self.object_properties_ = list(self.kb.object_properties)

        self.expression_encoder = ExpressionEncoder(device=device)

        self.data_obj_splits = self._get_split_catboost()
        self.boolean_dp = {}  # TODO

    def reset(self, seed=None, options=None):
        self.expression = self.init_expression()
        self.agents = copy(self.possible_agents)
        self.infos = {}
        self.cntr = 0
        self.reward = 0
        self.step_reward = 0
        self.nugyen_entropy = 0

        observations = {a: self._get_observation_state() for a in self.agents}

        infos = {a: {} for a in self.agents}

        return observations['a_0'], infos

    def get_theta(self, actions):
        return list(powerset(actions))[1:]

    def get_bpa(self, bpa):
        return softmax(bpa).flatten()

    def _get_masses(self, actions, bpa):

        theta = self.get_theta(actions)
        bpa = self.get_bpa(bpa)

        m1 = {}
        for i in range(len(theta)):
            m1[theta[i]] = bpa[i]
        return MassFunction(m1)

    def _get_best_action(self, m1, m2, singletons=True):
        combined = m1 & m2
        if singletons:
            best = 0.0
            best_action = frozenset({})
            for c in combined.all():
                if combined[c] > best:
                    best = combined[c]
                    best_action = c
            return list(best_action), nugyen_entropy(combined)
        else:
            return list(max(combined.pignistic(), key=combined.pignistic().get)), nugyen_entropy(combined)

    def _extract_actions(self, agent):
        c1 = agent["actions"][:2]
        c2 = agent["actions"][2:4]
        p = agent["actions"][4:6]
        bo = agent["actions"][6:8]
        q = agent["actions"][8:10]
        eq = agent["actions"][10:12]
        flags = agent["actions"][12:14]
        stop = agent["actions"][14:16]
        card = agent["actions"][16:]

        return c1, c2, p, bo, q, eq, flags, stop, card

    def _extract_bpa(self, agent):
        c1 = agent["masses"][0]
        c2 = agent["masses"][1]
        p = agent["masses"][2]
        bo = agent["masses"][3]
        q = agent["masses"][4]
        e = agent["masses"][5]
        flags = agent["masses"][6]
        stop = agent["masses"][7]
        card = agent["masses"][8]
        return c1, c2, p, bo, q, e, flags, stop, card

    def _get_observation_state(self):
        return self.expression_encoder.encode(str(self.expression))

    def step(self, actions):
        self.cntr += 1
        info = {}
        old_expression = self.expression

        c1_a1, c2_a1, p_a1, bo_a1, q_a1, e_a1, flags_a1, stop_a1, card_a1 = self._extract_actions(
            actions["a_0"]
        )
        c1_a2, c2_a2, p_a2, bo_a2, q_a2, e_a2, flags_a2, stop_a2, card_a2 = self._extract_actions(
            actions["a_1"]
        )

        bpa_a1_c1, bpa_a1_c2, bpa_a1_p, bpa_a1_bo, bpa_a1_q, bpa_a1_e, bpa_a1_flags, bpa_a1_stop, bpa_a1_card = (
            self._extract_bpa(actions["a_0"])
        )
        bpa_a2_c1, bpa_a2_c2, bpa_a2_p, bpa_a2_bo, bpa_a2_q, bpa_a2_e, bpa_a2_flags, bpa_a2_stop, bpa_a2_card = (
            self._extract_bpa(actions["a_1"])
        )

        c1_a1, c1_a2 = self.constrained_action_sample(c1_a1, c1_a2)

        c2_a1, c2_a2 = self.constrained_action_sample(c2_a1, c2_a2)

        p_a1, p_a2 = self.constrained_action_sample(p_a1, p_a2)

        bo_a1, bo_a2 = self.constrained_action_sample(bo_a1, bo_a2)

        q_a1, q_a2 = self.constrained_action_sample(q_a1, q_a2)

        e_a1, e_a2 = self.constrained_action_sample(e_a1, e_a2)

        flags_a1, flags_a2 = self.constrained_action_sample(flags_a1, flags_a2)

        stop_a1, stop_a2 = self.constrained_action_sample(stop_a1, stop_a2)

        card_a1, card_a2 = self.constrained_action_sample(card_a1, card_a2)

        c1_m1 = self._get_masses(c1_a1, bpa_a1_c1)
        c1_m2 = self._get_masses(c1_a2, bpa_a2_c1)
        best_action_1, a1_entropy = self._get_best_action(c1_m1, c1_m2)

        c2_m1 = self._get_masses(c2_a1, bpa_a1_c2)
        c2_m2 = self._get_masses(c2_a2, bpa_a2_c2)
        best_action_2, a2_entropy = self._get_best_action(c2_m1, c2_m2)

        p_m1 = self._get_masses(p_a1, bpa_a1_p)
        p_m2 = self._get_masses(p_a2, bpa_a2_p)
        best_action_3, a3_entropy = self._get_best_action(p_m1, p_m2)

        bo_m1 = self._get_masses(bo_a1, bpa_a1_bo)
        bo_m2 = self._get_masses(bo_a2, bpa_a2_bo)
        best_action_4, a4_entropy = self._get_best_action(bo_m1, bo_m2)

        q_m1 = self._get_masses(q_a1, bpa_a1_q)
        q_m2 = self._get_masses(q_a2, bpa_a2_q)
        best_action_5, a5_entropy = self._get_best_action(q_m1, q_m2)

        flags_m1 = self._get_masses(flags_a1, bpa_a1_flags)
        flags_m2 = self._get_masses(flags_a2, bpa_a2_flags)
        best_action_6, a6_entropy = self._get_best_action(flags_m1, flags_m2)

        stop_m1 = self._get_masses(stop_a1, bpa_a1_stop)
        stop_m2 = self._get_masses(stop_a2, bpa_a2_stop)
        best_action_7, a7_entropy = self._get_best_action(stop_m1, stop_m2)

        e_m1 = self._get_masses(e_a1, bpa_a1_e)
        e_m2 = self._get_masses(e_a2, bpa_a2_e)
        best_action_8, a8_entropy = self._get_best_action(e_m1, e_m2)

        card_m1 = self._get_masses(card_a1, bpa_a1_card)
        card_m2 = self._get_masses(card_a2, bpa_a2_card)
        best_action_9, a9_entropy = self._get_best_action(card_m1, card_m2)
        self.nugyen_entropy = - np.sum([a1_entropy, a2_entropy, a3_entropy, a4_entropy, a5_entropy, a6_entropy,
                                     a7_entropy, a8_entropy, a9_entropy])
        mass_functions = [
            c1_m1, c1_m2,
            c2_m1, c2_m2,
            p_m1, p_m2,
            bo_m1, bo_m2,
            q_m1, q_m2,
            flags_m1, flags_m2,
            stop_m1, stop_m2,
            e_m1, e_m2,
            card_m1, card_m2]

        pl_bel_differences = []
        for m in mass_functions:
            pl_bel_differences.append((np.array(list(m.pl().values())) - np.array(list(m.bel().values()))).mean())

        self.pl_bel_gaps = np.mean(pl_bel_differences)

        self.conflict = np.mean([c1_m1.conflict(c1_m2),
                                 c2_m1.conflict(c2_m2),
                                 p_m1.conflict(p_m2),
                                 bo_m1.conflict(bo_m2),
                                 q_m1.conflict(q_m2),
                                 flags_m1.conflict(flags_m2),
                                 stop_m1.conflict(stop_m2),
                                 e_m1.conflict(e_m2),
                                 card_m1.conflict(card_m2)])
        info['pl_bel_gaps'] = self.pl_bel_gaps
        info['conflict'] = self.conflict

        if (self.cntr == self.max_action) or (best_action_7[0] == 1):
            stop = True
        else:
            stop = False
        old_f1_score = evaluate_concept(
            self.kb, self.expression, F1(), self.lp.encode_kb(self.kb)
        ).q
        if stop:
            info["f1"] = old_f1_score
            terminations = truncations = {"a_0": True, "a_1": True}
            self.reset()
            # rewards = {"a_0": 0, "a_1": 0}

        if not stop:

            terminations = truncations = {"a_0": False, "a_1": False}
            set_operators = self.mappings["set"][best_action_4[0]]
            if best_action_6[0] == 0:
                if best_action_4[0] == 2:
                    self.expression = set_operators(self.expression)
                else:
                    self.expression = set_operators(
                        (
                            self.expression,
                            self.mappings["concepts"][best_action_1[0]],
                        )
                    )
            else:
                quantifier = self.mappings["quantifier"][best_action_5[0]]
                cardinality = self.mappings['cardinality'][best_action_8[0]]
                if self.expression == self.cg.thing:
                    if not isinstance(self.mappings["properties"][best_action_3[0]], OWLDataPropertyExpression):
                        if best_action_6[0] == 1:
                            self.expression = quantifier(
                                self.mappings["concepts"][best_action_1[0]],
                                self.mappings["properties"][best_action_3[0]],
                            )
                        if best_action_6[0] == 2:
                            self.expression = cardinality(
                                self.mappings["concepts"][best_action_1[0]],
                                self.mappings["properties"][best_action_3[0]],
                                best_action_9[0]
                            )
                    else:
                        if self.mappings["properties"][
                            best_action_3[0]] in self.boolean_dp:  # TODO: create booldp dictionary
                            value = list(self.kb.get_data_property_values(self.pos_example,
                                                                          self.boolean_dp[self.mappings["properties"][
                                                                              best_action_3[0].reminder]]))[0]
                            if (str(value).lower() == 'false') or (str(value).lower() == '0'):
                                data_has_value = OWLDataHasValue(self.mappings["properties"][best_action_3[0]],
                                                                 OWLLiteral(False))  # HEREEE
                            else:
                                data_has_value = OWLDataHasValue(self.mappings["properties"][best_action_3[0]],
                                                                 OWLLiteral(True))
                            if best_action_4[0] != 2:
                                self.expression = set_operators((self.expression, data_has_value))
                            else:
                                self.expression = set_operators(self.cg.union(self.expression, data_has_value))
                        else:
                            try:
                                if self.mappings["properties"][best_action_3[0]] in self.numeric_dp:

                                    value = self.numeric_dp[self.mappings["properties"][best_action_3[0]]]

                                    if (best_action_6[0] == 0) or (best_action_6[0] == 1):

                                        resteriction = 'min_inc'
                                    else:
                                        resteriction = 'max_inc'
                                    self.expression = create_numeric_data_some_value(
                                        self.mappings["properties"][best_action_3[0]],
                                        value, resteriction)
                            except:
                                pass
                else:
                    if not isinstance(self.mappings["properties"][best_action_3[0]], OWLDataPropertyExpression):
                        if best_action_6[0] == 1:
                            self.expression = quantifier(
                                self.expression,
                                self.mappings["properties"][best_action_3[0]],
                            )
                        if best_action_6[0] == 2:
                            self.expression = cardinality(
                                self.expression,
                                self.mappings["properties"][best_action_3[0]],
                                best_action_9[0]
                            )
                    else:
                        if self.mappings["properties"][
                            best_action_3[0]] in self.boolean_dp:  # TODO: create booldp dictionary
                            value = list(self.kb.get_data_property_values(self.pos_example,
                                                                          self.boolean_dp[self.mappings["properties"][
                                                                              best_action_3[0]].reminder]))[0]
                            if (str(value).lower() == 'false') or (str(value).lower() == '0'):
                                bool_exp = OWLDataHasValue(self.mappings["properties"][best_action_3[0]],
                                                           OWLLiteral(False))
                            else:
                                bool_exp = OWLDataHasValue(self.mappings["properties"][best_action_3[0]],
                                                           OWLLiteral(True))
                            self.expression = set_operators(
                                (
                                    self.expression,
                                    bool_exp))
                        if self.mappings["properties"][best_action_3[0]] in self.numeric_dp:

                            value = self.numeric_dp[self.mappings["properties"][
                                best_action_3[0]]]  # TODO: create numericdp dictionary
                            if value == None:
                                value = 0
                            if (best_action_6[0] == 0) or (best_action_6[0] == 1):
                                resteriction = 'min_inc'
                            else:
                                resteriction = 'max_inc'
                                numeric_exp = create_numeric_data_some_value(
                                    self.mappings["properties"][best_action_3[0]],
                                    value, resteriction)
                                if best_action_4[0] == 2:
                                    self.expression = set_operators(self.expression)
                                else:

                                    self.expression = set_operators((
                                        self.expression,
                                        numeric_exp))

            # reward = 0
            try:
                self.f1_score = evaluate_concept(self.kb, self.expression, F1(), self.lp.encode_kb(self.kb)).q
                info['f1'] = self.f1_score
                self.reward = self.f1_score
            except:
                self.expression = old_expression
                self.reward = -1
                # self.reward = self.f1_score + (1 / concept_len(self.expression))
            if (self.f1_score > 0.8) and (self.f1_score < 0.9):
                self.export_expr(
                    "80_90_Top_DL.txt", owl_expression_to_dl(self.expression) + '\n' + str(self.f1_score)
                )
            elif (self.f1_score > 0.9) and (self.f1_score < 0.95):
                self.export_expr(
                    "90_95_Top_DL.txt", owl_expression_to_dl(self.expression) + '\n' + str(self.f1_score)
                )
            elif self.f1_score > 95:
                self.export_expr("above_95_Top_DL.txt",
                                 owl_expression_to_dl(self.expression) + '\n' + str(self.f1_score))

            else:
                self.export_expr("below_80_Top_DL.txt",
                                 owl_expression_to_dl(self.expression) + '\n' + str(self.f1_score))

        self.rewards = {"a_0": self.reward, "a_1": self.reward}
        observations = {a: self._get_observation_state() for a in self.agents}

        infos = {'a_0': info, 'a_1': info}
        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, self.rewards, terminations, truncations, infos

    def constrained_action_sample(self, a1, a2):
        # should be okay since we are implementing a shared policy network for both agents for stability;
        # thus action logits will be obtained from the same distribution
        if len(set(a1).intersection(set(a2))) == 0:
            a1[0] = a2[0]
        return a1, a2

    def init_expression(self):
        candidate_exp = []
        direct_parent_conc = set()
        direct_parent_obj_prop = set()
        direct_parent_data_prop = set()

        for i in self.kb.concepts:
            conc_list = list(self.kb.reasoner.instances(i))
            self.pos_example = np.random.choice(list(self.pos))
            if self.pos_example in conc_list:
                direct_parent_conc.add(i)
                candidate_exp.append(i)
                if check_generators(self.kb.get_object_properties_for_ind(self.pos_example)) != False:
                    for x in self.kb.get_object_properties_for_ind(self.pos_example):
                        direct_parent_obj_prop.add(x)

                if check_generators(self.kb.get_data_properties_for_ind(self.pos_example)) != False:
                    for x in self.kb.get_data_properties_for_ind(self.pos_example):
                        direct_parent_data_prop.add(x)

                if check_generators(self.kb.get_boolean_data_properties()) != False:
                    for x in self.kb.get_boolean_data_properties():
                        if check_generators(self.kb.get_data_property_values(self.pos_example, x)) == False:
                            candidate_exp.append(OWLDataHasValue(x, OWLLiteral(False)))
                            candidate_exp.append(OWLDataHasValue(x, OWLLiteral(True)))
                        else:
                            value = list(self.kb.get_data_property_values(self.pos_example, x))[0]
                            candidate_exp.append(OWLDataHasValue(x, value))

        if len(direct_parent_conc) > 1:
            exp = self.kb.generator.union([x for x in direct_parent_conc])
            candidate_exp.append(exp)
            for i in direct_parent_obj_prop:
                candidate_exp.append(self.kb.generator.existential_restriction(exp, i))
                candidate_exp.append(self.kb.generator.universal_restriction(exp, i))
        else:
            if len(direct_parent_conc) >0:
                exp = self.kb.generator.existential_restriction([x for x in direct_parent_conc][0],
                                                            direct_parent_obj_prop.pop())
                n_props = np.random.randint(0, len(direct_parent_obj_prop) + 1)
            else:
                n_props = 0
            if n_props == 0:
                pass
            else:
                props = list(direct_parent_obj_prop)
                for i in range(n_props):
                    exp = self.kb.generator.existential_restriction(exp, props[i])
                    candidate_exp.append(exp)
                for i in range(n_props):
                    exp = self.kb.generator.universal_restriction(exp, props[i])
                    candidate_exp.append(exp)
        if n_props == 0:
            pass
        else:
            candidate_exp.append(exp)

        for i in self.kb.get_numeric_data_properties():
            try:
                value = self.numeric_dp[i]
                candidate_exp.append(create_numeric_data_some_value(i, value, 'min_inc'))
                candidate_exp.append(create_numeric_data_some_value(i, value, 'max_inc'))

            except:
                value = 1.0
                candidate_exp.append(create_numeric_data_some_value(i, value, 'min_inc'))
                candidate_exp.append(create_numeric_data_some_value(i, value, 'max_inc'))

        exp = candidate_exp[0]
        for i in candidate_exp:
            if evaluate_concept(self.kb, i, F1(), self.lp.encode_kb(self.kb)).q > evaluate_concept(self.kb, exp, F1(),
                                                                                                   self.lp.encode_kb(
                                                                                                           self.kb)).q:
                exp = i
        return np.random.choice([exp, self.cg.thing])

    def split_catboost(self, x, y):

        if np.issubdtype(x.dtype, np.number):

            x = np.array(x).reshape(-1, 1)
            y = np.array(y)
            try:
                pool = Pool(x, y)
                model = CatBoostClassifier(iterations=1, depth=1, logging_level='Silent')
                model.fit(pool)

                with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
                    tmp_path = tmp.name

                model.save_model(tmp_path, format='json')

                with open(tmp_path, 'r') as f:
                    model_json = json.load(f)
                global s
                s = model_json
                split_value = model_json['oblivious_trees'][0]['splits'][0]['border']
                os.remove(tmp_path)
                return split_value
            except:
                pass
                return
        else:
            return

    def _get_numeric_splits(self):
        dp_dict = dict(
            zip([x.iri.reminder for x in list(self.kb.get_data_properties())], list(self.kb.get_data_properties())))
        x, y = self._get_data_prop()

        dp_numeric = {}
        for i in x.columns:
            try:
                dp_numeric[dp_dict[x[i].name]] = self.split_catboost(x[i], y)
            except:
                pass
        return dp_numeric

    def render(self):
        print(self.expression.str)
        return

    def export_expr(self, file_name, value):
        with open(f"{file_name}.txt", "a", encoding="utf-8") as myfile:
            myfile.write(value + "\n\n")

    def _get_data_prop(self):
        pos_vals = defaultdict(list)
        neg_vals = defaultdict(list)

        for j in self.kb.get_data_properties():
            for i in self.pos:
                prop = list(self.kb.get_data_property_values(i, j))

                if len(prop) > 0:
                    for k in range(len(prop)):
                        try:
                            pos_vals[j.iri.reminder].append(float(prop[k].get_literal()))
                        except:
                            pos_vals[j.iri.reminder].append(prop[k].get_literal())

            for i in self.neg:
                prop = list(self.kb.get_data_property_values(i, j))
                if len(prop) > 0:
                    for k in range(len(prop)):
                        try:
                            neg_vals[j.iri.reminder].append(float(prop[k].get_literal()))
                        except:
                            neg_vals[j.iri.reminder].append(prop[k].get_literal())

        pos_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in pos_vals.items()]))

        pos_df['target'] = 1

        neg_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in neg_vals.items()]))

        neg_df['target'] = 0

        df = pd.concat([pos_df, neg_df], axis=0)
        df = df.fillna('nan')

        x = df.iloc[:, :-1]
        y = df['target']

        return x, y

    def _get_split_catboost(self):
        x, y = self._get_data_prop()
        data_prop_splits = {}
        for i in x.select_dtypes(include='number').columns:
            data_prop_splits[i] = self.split_catboost(x[i], y)

        return data_prop_splits

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(low=-1, high=1, shape=(1, 128)
                          )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Dict(
            {
                "actions": spaces.MultiDiscrete(
                    [
                        len(self.concepts),
                        len(self.concepts),

                        len(self.concepts),
                        len(self.concepts),

                        len(self.properties),
                        len(self.properties),

                        len(list(self.set_operators)),
                        len(list(self.set_operators)),

                        len(list(self.quantifiers)),
                        len(list(self.quantifiers)),

                        len(list(self.cardinality)),
                        len(list(self.cardinality)),

                        3,  # flag
                        3,  # flag

                        2,  # stopping
                        2,  # stopping

                        10,
                        10
                    ]
                ),
                "masses": spaces.Box(0, 1, (9, 15))})
