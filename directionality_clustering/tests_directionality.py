from direction_validation import DirectionValidation
from unittest import TestCase
from data.verbs import *


def get_avg_of_multi_worded_queries_given_input_tokens(assert_value, split_multi_word_token, index,
                                                       obj_direction_validation,partner_token,input_tokens,all_verbs):
    if index == 0:
        flag_multi_word_token_goes_first = True
    else:
        flag_multi_word_token_goes_first = False


    avg = obj_direction_validation.give_verb_types_return_multi_word_query_averages(split_multi_word_token, flag_multi_word_token_goes_first,
        partner_token, all_verbs)


    assert avg == assert_value

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
        prob = self.obj_direction_validation.find_prob_multiple_tokens_per_verb(verb, flag_multi_word_token_goes_first, left_token,
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



    def test_find_per_adverb_prob_education_stability(self):
            mlm_first_query_token="education"
            mlm_causal_causal_adverb="improves"
            mlm_second_query_token="stability"
            sequence = str(mlm_first_query_token) + " " + (
                mlm_causal_causal_adverb) + " " + self.obj_direction_validation.tokenizer.mask_token
            token_probs, sequence = self.obj_direction_validation.create_prob_dict(sequence)
            prob= token_probs[mlm_second_query_token]
            assert prob == 0.0004211414197925478

    def test_get_avg_of_multi_worded_queries_for_promotes(self):

        input_tokens=["rice production","income"]
        split_multi_word_token=["rice","production"]
        assert_value=0.0001023119330056943
        index=0
        partner_token="income"
        get_avg_of_multi_worded_queries_given_input_tokens(assert_value,split_multi_word_token,index,self.obj_direction_validation,partner_token,input_tokens,all_promote_verbs)


        input_tokens = ["income","rice production"]
        split_multi_word_token = ["rice", "production"]
        #index must be 1 when the multiwordtoken comes second. e.g.;, income promotes rice production
        index = 1
        assert_value = 0.03647934375840123
        partner_token = "income"
        get_avg_of_multi_worded_queries_given_input_tokens(assert_value, split_multi_word_token, index,
                                                           self.obj_direction_validation,partner_token,input_tokens,all_promote_verbs)

        input_tokens = ["rice production stability", "income"]
        split_multi_word_token = ["rice", "production","stability"]
        assert_value = 7.500293637955717e-05
        index = 0
        partner_token = "income"
        get_avg_of_multi_worded_queries_given_input_tokens(assert_value, split_multi_word_token, index,
                                                           self.obj_direction_validation,partner_token,input_tokens,all_promote_verbs)

        input_tokens = ["income", "rice production stability"]
        split_multi_word_token = ["rice", "production","stability"]
        # index must be 1 when the multiwordtoken comes second. e.g.;, income promotes rice production
        index = 1
        assert_value = 0.024320398707156226
        partner_token = "income"
        get_avg_of_multi_worded_queries_given_input_tokens(assert_value, split_multi_word_token, index,
                                                           self.obj_direction_validation, partner_token, input_tokens,
                                                           all_promote_verbs)


    def test_get_avg_of_multi_worded_queries_for_inhibits(self):
        ###tests for inhibit related queries
        input_tokens = ["rice production", "income"]
        split_multi_word_token = ["rice", "production"]
        assert_value = 7.310948785743676e-05
        index = 0
        partner_token = "income"
        get_avg_of_multi_worded_queries_given_input_tokens(assert_value, split_multi_word_token, index,
                                                           self.obj_direction_validation, partner_token, input_tokens,
                                                           all_inhibits_verbs)



        input_tokens = ["income","rice production"]
        split_multi_word_token = ["rice", "production"]
        #index must be 1 when the multiwordtoken comes second. e.g.;, income promotes rice production
        index = 1
        assert_value = 0.03481978230411187
        partner_token = "income"
        get_avg_of_multi_worded_queries_given_input_tokens(assert_value, split_multi_word_token, index,
                                                           self.obj_direction_validation,partner_token,input_tokens,all_inhibits_verbs)


        input_tokens = ["rice production stability", "income"]
        split_multi_word_token = ["rice", "production", "stability"]
        assert_value = 9.413996061387782e-05
        index = 0
        partner_token = "income"
        get_avg_of_multi_worded_queries_given_input_tokens(assert_value, split_multi_word_token, index,
                                                           self.obj_direction_validation, partner_token, input_tokens,
                                                           all_inhibits_verbs)


        input_tokens = ["income", "rice production stability"]
        split_multi_word_token = ["rice", "production","stability"]
        # index must be 1 when the multiwordtoken comes second. e.g.;, income promotes rice production
        index = 1
        assert_value = 0.023213260299703126
        partner_token = "income"
        get_avg_of_multi_worded_queries_given_input_tokens(assert_value, split_multi_word_token, index,
                                                           self.obj_direction_validation, partner_token, input_tokens,
                                                           all_inhibits_verbs)


        input_tokens = ["main crop", "income"]
        split_multi_word_token = ["main", "crop"]
        # index must be 1 when the multiwordtoken comes second( e.g.;, income promotes rice production) else zero
        index = 0
        assert_value = 0.0006208729610079899
        partner_token = "income"
        get_avg_of_multi_worded_queries_given_input_tokens(assert_value, split_multi_word_token, index,
                                                           self.obj_direction_validation, partner_token, input_tokens,
                                                           all_does_not_promote_verbs)


        input_tokens = ["main crop", "income"]
        split_multi_word_token = ["main", "crop"]
        # index must be 1 when the multiwordtoken comes second( e.g.;, income promotes rice production) else zero
        index = 0
        assert_value = 0.00022078755137044936
        partner_token = "income"
        get_avg_of_multi_worded_queries_given_input_tokens(assert_value, split_multi_word_token, index,
                                                           self.obj_direction_validation, partner_token, input_tokens,
                                                           all_does_not_inhibits_verbs)


    def test_get_single_word_queries_for_does_not(self):
        input_tokens = ["education", "stability"]
        mlm_causal_causal_adverb = "does not improve"
        sequence = str(input_tokens[0]) + " " + (
            mlm_causal_causal_adverb) + " " + self.obj_direction_validation.tokenizer.mask_token
        prob_dict, sequence = self.obj_direction_validation.create_prob_dict(sequence)
        print(prob_dict['stability'])
        assert prob_dict['stability'] == 2.5620288397476543e-06

        input_tokens = ["education", "stability"]
        mlm_causal_causal_adverb = "does not diminish"
        sequence = str(input_tokens[0]) + " " + (
            mlm_causal_causal_adverb) + " " + self.obj_direction_validation.tokenizer.mask_token
        prob_dict, sequence = self.obj_direction_validation.create_prob_dict(sequence)
        print(prob_dict['stability'])
        assert prob_dict['stability'] == 4.7516396080027334e-06


    #average of all verbs in a class. eg. average of education improves stability, education promotes stability etc
    def test_find_average_prob_mlm_single_word_tokens(self):
        input_tokens = ["education", "stability"]
        avg = self.obj_direction_validation.find_average_causal_mlm(all_promote_verbs, input_tokens)
        assert avg == 0.00019201972463633865

        input_tokens = ["education", "stability"]
        avg = self.obj_direction_validation.find_average_causal_mlm(all_inhibits_verbs, input_tokens)
        assert avg == 0.00012041754234815016


        input_tokens = ["education", "stability"]
        avg = self.obj_direction_validation.find_average_causal_mlm(all_does_not_promote_verbs, input_tokens)
        assert avg == 0.00014735285367351025

        input_tokens = ["education", "stability"]
        avg = self.obj_direction_validation.find_average_causal_mlm(all_does_not_inhibits_verbs, input_tokens)
        assert avg == 8.410893497057259e-05
