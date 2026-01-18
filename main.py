from utils.utils import *
from Lyra.lyra_agent import Lyra

from ontolearn.learners import Drill, CELOE
from ontolearn.concept_learner import EvoLearner
import random
import numpy as np
import torch

Drill.name = 'Drill'


if __name__ == '__main__':

    KGs = [r"...\sml\carc\carcinogenesis.owl"]
    lps = [r"...\sml\carc"]
    print('==============================================================================')
    rnd_seed = random.sample(range(1, 1000), 30)
    old_seeds = []
    for idx, i in enumerate(KGs):
        random.seed(rnd_seed[idx])
        np.random.seed(rnd_seed[idx])
        torch.manual_seed(rnd_seed[idx])
        print('random_seed: ', rnd_seed[idx])
        print('==============================================================================')
        print('Learning: ', i.split('\\')[-1].split('.')[0])
        pos, neg = pos_and_neg_parser(lps[idx])
        experiment_report('carc', [Lyra, EvoLearner, CELOE, Drill], 30,
                          i, pos, neg)
