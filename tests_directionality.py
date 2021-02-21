from direction_validation import DirectionValidation
from unittest import TestCase
from data.verbs import *

class tests_directionality(TestCase):
    input_tokens = ["rice production", "income"]
    obj_direction_validation = DirectionValidation("distilbert-base-uncased")

    def test_create_prob_dict(self):
            prob_dict,sequence=self.obj_direction_validation.create_prob_dict("rice production", "improves")
            print(prob_dict['income'])
            assert prob_dict['income'] == 7.95996020315215e-05

    def test_find_average_causal_mlm(self):
            avg = self.obj_direction_validation.find_average_causal_mlm(all_promote_adverbs, self.input_tokens)
            assert avg == 0.00013738061534240842
