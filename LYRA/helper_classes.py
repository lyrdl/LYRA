from collections import  defaultdict


class Hypothesis_Cache:
    def __init__(self, n):
        self.n = n
        self.hypo_dict = defaultdict(list)

    def _add_hypo(self, expression, score):
        best_score = 0
        if score > best_score:
            self.hypo_dict['exp'].append(expression)
            self.hypo_dict['score'].append(score)

        if len(self.hypo_dict) > 0:
            combined = list(zip(self.hypo_dict['exp'],
                                self.hypo_dict['score']))
            sorted_hypothesis = sorted(combined, key=lambda x: x[1], reverse=True)
            if len(sorted_hypothesis) > self.n:
                del sorted_hypothesis[-1]
            sorted_exp, sorted_score = zip(*sorted_hypothesis)

            self.hypo_dict['exp'] = list(sorted_exp)
            self.hypo_dict['score'] = list(sorted_score)
