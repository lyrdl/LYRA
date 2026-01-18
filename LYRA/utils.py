from itertools import chain, combinations
import torch

from ontolearn.knowledge_base import KnowledgeBase

from ontolearn.utils.static_funcs import concept_len
from ontolearn.metrics import F1
from ontolearn.quality_funcs import evaluate_concept
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import OWLNamedIndividual
from owlapy import owl_expression_to_dl
import json
from ontolearn.heuristics import CELOEHeuristic
from ontolearn.learners import CELOE

import numpy as np

from ontolearn.refinement_operators import ModifiedCELOERefinement

from collections import defaultdict

import pandas as pd
import time
from owlapy.class_expression import OWLDataSomeValuesFrom

from owlapy.owl_literal import OWLLiteral

from owlapy.owl_datatype import OWLDatatype
from owlapy.iri import IRI
from owlapy.class_expression import OWLFacetRestriction, OWLFacet, OWLDatatypeRestriction

def nugyen_entropy(m):
    all_hypothesis = []
    for i in m:
        all_hypothesis.append(m[i] * np.log2(m[i]))
    return np.sum(all_hypothesis)


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def pos_and_neg_parser(lp_path):
    pos_file = lp_path + r'/pos.txt'
    neg_file = lp_path + r'/neg.txt'

    with open(pos_file, encoding='utf-8') as file:
        pos_data = file.readlines()

    with open(neg_file, encoding='utf-8') as file:
        neg_data = file.readlines()

    pos = set([OWLNamedIndividual(x) for x in [x.replace('\n', '') for x in pos_data][:-1]])
    neg = set([OWLNamedIndividual(x) for x in [x.replace('\n', '') for x in neg_data]])
    return pos, neg


def experiment_report(lp_name, models, n_trials, original_g, pos, neg):
    kb = KnowledgeBase(path=original_g)

    lp = PosNegLPStandard(pos=pos, neg=neg)
    lp.all = pos.union(neg)

    results_dict = defaultdict(list)

    for m in models:
        for i in range(n_trials):
            start_time = time.time()
            if 'CLIP' == m.name:
                model = m(knowledge_base=kb, refinement_operator=ModifiedCELOERefinement(kb),
                          predictor_name='SetTransformer').get_length_predictor()
            elif 'celoe_python' == m.name:
                qual = F1()
                heur = CELOEHeuristic(expansionPenaltyFactor=0.05, startNodeBonus=1.0, nodeRefinementPenalty=0.01)
                op = ModifiedCELOERefinement(knowledge_base=kb, use_negation=False, use_all_constructor=False)
                model = CELOE(knowledge_base=kb,
                              max_runtime=600,
                              refinement_operator=op,
                              quality_func=qual,
                              heuristic_func=heur,
                              max_num_of_concepts_tested=100,
                              iter_bound=100)

            elif 'Lyra' == m.name:
                model = m(original_g, pos, neg)
            else:
                model = m(kb)

            if 'Lyra' == m.name:
                model.fit()
            else:
                model.fit(lp)

            end_time = time.time()
            if model.name != 'Lyra':
                f1 = evaluate_concept(kb, model.best_hypotheses(), F1(), lp.encode_kb(kb)).q
                con_len = concept_len(model.best_hypotheses())
                expr = owl_expression_to_dl(model.best_hypotheses())
                results_dict[model.name].append({'trial': i,
                                                 'f1': f1,
                                                 'time': end_time - start_time,
                                                 'concept_len': con_len,
                                                 'DL': expr})
            else:
                f1 = evaluate_concept(kb, model.best_hypotheses()[0], F1(), lp.encode_kb(kb)).q
                con_len = concept_len(model.best_hypotheses()[0])
                expr = owl_expression_to_dl(model.best_hypotheses()[0])
                results_dict[model.name].append({'trial': i,
                                                 'f1': f1,
                                                 'time': end_time - start_time,
                                                 'concept_len': con_len,
                                                 'DL': expr})

            print('trial: ', i, 'model name: ', model.name, 'f1: ', f1, 'time: ', np.round(end_time - start_time, 3),
                  'concept_len: ', con_len, 'expression: ', expr)
            print('=========================================================================')
            if model.name == 'Lyra':
                best_hypo = model.best_hypotheses()
            else:
                best_hypo = list(set(model.best_hypotheses(n=20)))
            for i in best_hypo:
                print('f1: ', evaluate_concept(kb, i, F1(), lp.encode_kb(kb)).q, '/n',
                      'exp:', owl_expression_to_dl(i))

    report_dict = {}
    for k, v in results_dict.items():
        f1 = []
        len_ = []
        time_ = []
        for i in v:
            f1.append(i['f1'])
            len_.append(i['concept_len'])
            time_.append(i['time'])
        report_dict[k] = {'models': ['EvoLearner', 'Lyra', 'Drill'],
                          'f1_mean': np.round(np.array(f1).mean(), 5),
                          'f1_std': np.round(np.array(f1).std(), 5),
                          'time': np.round(np.array(time_).mean(), 5),
                          'time_std': np.round(np.array(time_).std(), 5),
                          'DL_len': np.array(len_).mean()}
    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv(f"{lp_name}_out.csv", index=False)

    with open(f"{lp_name}_exp_results.csv", 'w', encoding="utf-8") as fp:
        json.dump(results_dict, fp)


def create_numeric_data_some_value(datatype, value, resteriction):
    double_datatype = OWLDatatype(IRI.create('http://www.w3.org/2001/XMLSchema#double'))
    literal = OWLLiteral(value)
    if resteriction == 'min_inc':
        facet_restriction = OWLFacetRestriction(OWLFacet.MIN_INCLUSIVE, literal)

    if resteriction == 'max_inc':
        facet_restriction = OWLFacetRestriction(OWLFacet.MAX_INCLUSIVE, literal)

    datatype_restriction = OWLDatatypeRestriction(double_datatype, [facet_restriction])

    d = OWLDataSomeValuesFrom(datatype, datatype_restriction)

    return d


def check_generators(iterable):
    try:
        next(iterable)
    except StopIteration:
        return False
    return
