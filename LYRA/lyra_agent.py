import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from envs.MADRL_env import Concept_Dec_POMPD
from utils.utils import *
from helper_classes.helper_classes import Hypothesis_Cache

class Lyra(nn.Module):
    name = 'Lyra'

    def __init__(self, knoweldge_base, pos, neg, run_name='R1', n_hypothesis=5):
        super().__init__()
        self.multi_a_env = Concept_Dec_POMPD(knoweldge_base, pos, neg, "f1",
                                             10)

        self.out_dim_cat = self.multi_a_env.action_space('a_0')['actions'].nvec.sum()
        self.out_dim_cont = np.prod(self.multi_a_env.action_space('a_0')['masses'].shape)
        self.network = nn.Sequential(
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
        )
        self.run_name = run_name
        self.cat_actor = layer_init(nn.Linear(64, self.out_dim_cat), std=0.01)
        self.cont_actor = layer_init(nn.Linear(self.out_dim_cat,
                                               self.out_dim_cont), std=0.01)

        self.critic = layer_init(nn.Linear(64, 1), std=1)

        torch.backends.cudnn.deterministic = False
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = 'cpu'
        self.log_std = nn.Parameter(torch.zeros(self.out_dim_cont))
        # self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.session_info = defaultdict(list)
        self.hypothesis = Hypothesis_Cache(n=n_hypothesis)

    def get_value(self, x):
        x = x.clone()
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):

        hidden = self.network(x)
        cat_action_logits = self.cat_actor(hidden)
        cont_action_logits = self.cont_actor(cat_action_logits)

        cat_actions_dim = self.multi_a_env.action_space("a_0")["actions"].nvec
        masses_action_dim = self.multi_a_env.action_space("a_0")["masses"].shape

        is_batch = cat_action_logits.dim() == 2
        if not is_batch:
            action_std = self.log_std.exp()

            bpa_props = Normal(
                cont_action_logits.reshape(
                    masses_action_dim
                ),
                action_std.reshape(masses_action_dim),
            )

        else:
            action_std = self.log_std.exp()
            bpa_props = Normal(loc=cont_action_logits.view(
                (cont_action_logits.size(0), masses_action_dim[0] * masses_action_dim[1])), scale=action_std)

        splits = torch.split(
            cat_action_logits,
            cat_actions_dim.tolist(),
            dim=(0 if not is_batch else 1),
        )
        cat_probs = [Categorical(logits=s) for s in splits]

        if action is None:
            cat_actions_a1 = torch.stack(
                [dist.sample() for dist in cat_probs], dim=1 if is_batch else 0
            )
            cat_actions_a2 = torch.stack(
                [dist.sample() for dist in cat_probs], dim=1 if is_batch else 0
            )
            masses_action_a1 = bpa_props.sample()
            masses_action_a2 = bpa_props.sample()
            log_propbs_cat_a_1 = torch.stack(
                [
                    cat_probs[x].log_prob(cat_actions_a1[..., x])
                    for x in range(len(cat_probs))
                ],
                dim=-1,
            ).sum(-1)
            log_propbs_cat_a_2 = torch.stack(
                [
                    cat_probs[x].log_prob(cat_actions_a2[..., x])
                    for x in range(len(cat_probs))
                ],
                dim=-1,
            ).sum(-1)
            if not is_batch:
                action = {
                    "a_0": {
                        "actions": cat_actions_a1,
                        "masses": masses_action_a1.reshape(*masses_action_dim),
                    },
                    "a_1": {
                        "actions": cat_actions_a2,
                        "masses": masses_action_a2.reshape(*masses_action_dim),
                    },
                }
        else:
            (
                cat_actions_a1,
                cat_actions_a2,
                masses_action_a1,
                masses_action_a2,
            ) = torch.split(
                action, [18, 18, 135, 135], dim=-1
            )  # TODO: Better splitting

            log_propbs_cat_a_1 = torch.stack(
                [
                    cat_probs[x].log_prob(cat_actions_a1[..., x])
                    for x in range(len(cat_probs))
                ],
                dim=-1,
            ).sum(-1)
            log_propbs_cat_a_2 = torch.stack(
                [
                    cat_probs[x].log_prob(cat_actions_a2[..., x])
                    for x in range(len(cat_probs))
                ],
                dim=-1,
            ).sum(-1)
        bpa_log_props_a1 = (
            bpa_props.log_prob(masses_action_a1)
            .sum(-1)
        )
        bpa_log_probs_a2 = (
            bpa_props.log_prob(masses_action_a2)
            .sum(-1)
        )

        if is_batch:
            aggregated_log_probs = (log_propbs_cat_a_1.sum(-1) + log_propbs_cat_a_2.sum(-1) + bpa_log_props_a1.sum(
                -1) + bpa_log_probs_a2.sum(-1))
        else:
            aggregated_log_probs = (
                    log_propbs_cat_a_1
                    + log_propbs_cat_a_2
                    + bpa_log_props_a1.sum(-1)
                    + bpa_log_probs_a2.sum(-1)
            )

        total_entropy = self.multi_a_env.nugyen_entropy

        if not is_batch:
            action_for_buffer = torch.cat(
                [
                    cat_actions_a1,
                    cat_actions_a2,
                    masses_action_a1.flatten(
                        start_dim=-(len(masses_action_dim))
                    ).view(-1),
                    masses_action_a2.flatten(
                        start_dim=-(len(masses_action_dim))
                    ).view(-1),
                ],
                dim=-1,
            )

        else:
            action_for_buffer = torch.cat(
                [
                    cat_actions_a1.squeeze(1),
                    cat_actions_a2.squeeze(1),
                    masses_action_a1.reshape(
                        -1, (masses_action_dim[0] * masses_action_dim[1])
                    ),
                    masses_action_a2.reshape(
                        -1, (masses_action_dim[0] * masses_action_dim[1])
                    ),
                ],
                dim=-1,
            )

        return (
            action,
            aggregated_log_probs,
            total_entropy,
            self.critic(hidden),
            action_for_buffer,
        )

    def best_hypotheses(self):
        return self.hypothesis.hypo_dict['exp']

    def fit(self, gamma=0.99, gae_lambda=0.95, batch_size=10, minibatch_size=32, ent_coef=0.01, vf_coef=0.5,
            target_kl=None,
            update_epochs=4, n_steps=12000, learning_rate=0.001, anneal_lr=True, n_hypothesis=20):

        print('Start learning...')

        optimizer = optim.Adam(self.parameters(), lr=learning_rate, eps=1e-5)
        global_step = 0
        self.start_time = time.time()

        next_obs, info = self.multi_a_env.reset()
        next_obs = next_obs.to(self.device)

        next_termination = torch.zeros(1).to(self.device)
        next_truncation = torch.zeros(1).to(self.device)
        num_updates = n_steps // batch_size
        obs = torch.zeros((n_steps, 64)).to(self.device)
        actions = torch.zeros((n_steps, 306)).to(self.device)  # TODO: subistitute the hardcoded)

        logprobs = torch.zeros((n_steps, 1)).to(self.device)
        rewards = torch.zeros((n_steps, 1)).to(self.device)
        terminations = torch.zeros((n_steps, 1)).to(self.device)
        truncations = torch.zeros((n_steps, 1)).to(self.device)
        values = torch.zeros((n_steps, 1)).to(self.device)

        for update in range(1, num_updates + 1):
            if anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * learning_rate
                optimizer.param_groups[0]["lr"] = lrnow
            for step in range(0, n_steps):
                global_step += 1
                obs[step] = next_obs
                terminations[step] = next_termination
                truncations[step] = next_truncation
                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value, action_for_buffer = (
                        self.get_action_and_value(next_obs.detach())
                    )
                    values[step] = value.flatten()
                actions[step] = action_for_buffer
                logprobs[step] = logprob
                action['a_0']['actions'] = action['a_0']['actions'].numpy()
                action['a_0']['masses'] = action['a_0']['masses'].numpy()
                action['a_1']['actions'] = action['a_1']['actions'].numpy()
                action['a_1']['masses'] = action['a_1']['masses'].numpy()

                next_obs, reward, termination, truncation, info = self.multi_a_env.step(action)
                self.hypothesis._add_hypo(self.multi_a_env.expression, self.multi_a_env.f1_score)
                self.conflict = self.multi_a_env.conflict
                self.pl_bel_gap = self.multi_a_env.pl_bel_gaps

                self.session_info['conflict'].append(self.conflict)
                self.session_info['pl_bl_gap'].append(self.pl_bel_gap)
                self.session_info['reward'].append(reward)

                rewards[step] = (torch.tensor(reward["a_0"]).to(self.device).view(-1))

                next_obs = self.multi_a_env._get_observation_state().to(self.device)
                next_termination = torch.Tensor([1 if termination["a_0"] else 0]).to(self.device)
                next_truncation = torch.Tensor([1 if termination["a_0"] else 0]).to(self.device)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.get_value(next_obs.detach()).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                next_done = torch.maximum(next_termination, next_truncation)
                dones = torch.maximum(terminations, truncations)
                for t in reversed(range(n_steps)):
                    if t == n_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                            rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                    )
                    advantages[t] = lastgaelam = (
                            delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + (1, 64))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + (1, 306))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            # Optimizing the policy and value network
            b_inds = np.arange(batch_size)
            for epoch in range(update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    optimizer.zero_grad()
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]
                    _, newlogprob, entropy, newvalue, action_for_buffer = (
                        self.get_action_and_value(
                            b_obs[mb_inds].squeeze(1).detach(), b_actions[mb_inds].detach()
                        )
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                    mb_advantages = b_advantages[mb_inds]
                    norm_adv = True
                    if norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                                mb_advantages.std() + 1e-8
                        )
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss = pg_loss1.mean()
                    # Value loss
                    newvalue = newvalue.view(-1)
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
                    loss.backward()
                    optimizer.step()

                if target_kl is not None:
                    if approx_kl > target_kl:
                        break

        report_df = pd.DataFrame(self.session_info).T
        report_df.to_csv("session_out.csv", index=False)
        print("SPS:", int(global_step / (time.time() - self.start_time)))
