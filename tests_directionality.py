from direction_validation import DirectionValidation
from unittest import TestCase
from data.verbs import *

class tests_directionality(TestCase):

    obj_direction_validation = DirectionValidation("distilbert-base-uncased")

    def test_create_prob_dict(self):
            input_tokens = ["rice production", "income"]
            mlm_causal_causal_adverb="improves"
            sequence = str(input_tokens[0]) + " " + (mlm_causal_causal_adverb) + " " + self.obj_direction_validation.tokenizer.mask_token
            prob_dict,sequence=self.obj_direction_validation.create_prob_dict(sequence)
            print(prob_dict['income'])
            assert prob_dict['income'] == 7.95996020315215e-05

    def test_find_prob_multiple_tokens_per_verb(self):
        verb="boosts"
        flag_multi_word_token_goes_first=True
        left_token="rice"
        all_tokens_in_between="production"
        query_token="income"
        prob = self.find_prob_multiple_tokens_per_verb(verb, flag_multi_word_token_goes_first, left_token,
                                                       all_tokens_in_between, query_token)
        assert prob==0.0003901408636011183

    def test_find_average_causal_mlm_rice_income(self):
            input_tokens = ["rice", "income"]
            verb="improves"
            prob = self.obj_direction_validation.find_prob_for_each_query(verb, input_tokens)
            assert prob == 8.142540900735185e-05


    def test_find_average_causal_mlm_rice_production_income(self):
            input_tokens = ["rice production", "income"]
            avg = self.obj_direction_validation.find_average_causal_mlm(all_promote_verbs, input_tokens)
            assert avg == 0.00013738061534240842
            avg = self.obj_direction_validation.find_average_causal_mlm(all_inhibits_verbs, input_tokens)
            assert avg == 9.93180219666101e-05
            avg = self.obj_direction_validation.find_average_causal_mlm(all_causal_verbs, input_tokens)
            assert avg ==0.0003280365199316293

    def test_find_average_causal_mlm_education_stability(self):
            input_tokens = ["education", "stability"]
            avg = self.obj_direction_validation.find_average_causal_mlm(all_promote_verbs, input_tokens)
            assert avg == 0.00019201972463633865

    def test_find_per_adverb_prob_education_stability(self):
            mlm_first_query_token="education"
            mlm_causal_causal_adverb="improves"
            mlm_second_query_token="stability"
            sequence = str(mlm_first_query_token) + " " + (
                mlm_causal_causal_adverb) + " " + self.obj_direction_validation.tokenizer.mask_token
            token_probs, sequence = self.obj_direction_validation.create_prob_dict(sequence)
            prob= token_probs[mlm_second_query_token]
            assert prob == 0.0004211414197925478

    def test_get_avg_of_multi_worded_queries(self):
        input_tokens=["rice production","income"]
        split_multi_word_token=["rice","production"]
        avg=self.obj_direction_validation.get_avg_of_multi_worded_queries(0,len(input_tokens),input_tokens,split_multi_word_token)
        assert avg==0.0001023119330056943

