from direction_validation import DirectionValidation
from unittest import TestCase
from data.verbs import *

class tests_directionality(TestCase):

    obj_direction_validation = DirectionValidation("distilbert-base-uncased")

    def test_create_prob_dict(self):
            prob_dict,sequence=self.obj_direction_validation.create_prob_dict("rice production", "improves")
            print(prob_dict['income'])
            assert prob_dict['income'] == 7.95996020315215e-05

    def test_find_average_causal_mlm_rice_production_income(self):
            input_tokens = ["rice production", "income"]
            avg = self.obj_direction_validation.find_average_causal_mlm(all_promote_adverbs, input_tokens)
            assert avg == 0.00013738061534240842
            avg = self.obj_direction_validation.find_average_causal_mlm(all_inhibits_adverbs, input_tokens)
            assert avg == 9.93180219666101e-05
            avg = self.obj_direction_validation.find_average_causal_mlm(all_causal_adverbs, input_tokens)
            assert avg ==0.0003280365199316293

    def test_find_average_causal_mlm_education_stability(self):
            input_tokens = ["education", "stability"]
            avg = self.obj_direction_validation.find_average_causal_mlm(all_promote_adverbs, input_tokens)
            assert avg == 0.00019201972463633865

    def test_find_per_adverb_prob_education_stability(self):
            mlm_first_query_token="education"
            mlm_causal_causal_adverb="improves"
            mlm_second_query_token="stability"
            token_probs, sequence = self.obj_direction_validation.create_prob_dict(mlm_first_query_token, mlm_causal_causal_adverb)
            prob= token_probs[mlm_second_query_token]
            assert prob == 0.0004211414197925478
            
