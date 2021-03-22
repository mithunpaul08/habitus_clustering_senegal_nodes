from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from glove_read_get_embed import read_eidos_stopwords
import sys
import nltk
import torch.nn.functional as F
from nltk.corpus import stopwords
from transformers import pipeline
from pprint import pprint
from utils import *
import torch
from data.verbs import *
import os
from datetime import date
import logging

LOGGING_LEVEL="logging.INFO"
VARIABLES_FILE="data/query_directionality_variables.csv"
LIST_MODEL_NAME=["distilbert-base-uncased"]
OUTPUT_FILE="outputs/prob.json"

class DirectionValidation:

    def __init__(self,model_name, input_tokens):

        self.logger = logging.getLogger(__name__)
        today = date.today()
        today_date=today.strftime("%b-%d-%Y")
        log_file_name = "".join(input_tokens[0].split())+"_"+"".join(input_tokens[1].split())+"_"+model_name+today_date+".log"
        full_path=os.path.join("./logs",log_file_name)
        FORMAT='%(message)s'

        logging.basicConfig(
            format=FORMAT,
            level=logging.INFO,
            filename=full_path,
            filemode='w'
        )
        self.logger.info(f"modelname={model_name}")

        self.MLM_MODEL=model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.MLM_MODEL)
        self.model = AutoModelForMaskedLM.from_pretrained(self.MLM_MODEL)

    def create_prob_dict(self,sequence):
        input = self.tokenizer.encode(sequence, return_tensors="pt")
        mask_token_index = torch.where(input == self.tokenizer.mask_token_id)[1]
        token_logits = self.model(input).logits
        mask_token_logits = token_logits[0, mask_token_index, :]

        mask_token_logits = F.softmax(mask_token_logits)
        all_token_probs = torch.topk(mask_token_logits, len(self.tokenizer.get_vocab()), dim=1)
        alltoken_indices = all_token_probs[1]
        alltoken_probs = all_token_probs[0]
        token_probs = {}
        for token_index, prob in zip(alltoken_indices[0], all_token_probs[0][0]):
            token = self.tokenizer.decode(token_index)
            token_probs[token] = prob
        return token_probs,sequence



    def create_prob_dict_two_strings(self,mlm_first_query_token,mlm_causal_causal_verb,mlm_second_query_token,does_not=""):
        sequence = str(mlm_first_query_token) +" "+ does_not+" " + str(mlm_causal_causal_verb) + " "+ str(mlm_second_query_token)+" " + self.tokenizer.mask_token
        input = self.tokenizer.encode(sequence, return_tensors="pt")
        mask_token_index = torch.where(input == self.tokenizer.mask_token_id)[1]
        token_logits = self.model(input).logits
        mask_token_logits = token_logits[0, mask_token_index, :]

        mask_token_logits = F.softmax(mask_token_logits)
        all_token_probs = torch.topk(mask_token_logits, len(self.tokenizer.get_vocab()), dim=1)
        alltoken_indices = all_token_probs[1]
        alltoken_probs = all_token_probs[0]
        token_probs = {}
        for token_index, prob in zip(alltoken_indices[0], all_token_probs[0][0]):
            token = self.tokenizer.decode(token_index)
            token_probs[token] = prob
        return token_probs,sequence

    def find_average_causal_mlm_given_input_tokens_as_string(self,list_verbs, mlm_first_query_token,mlm_second_query_token,mlm_token_to_check,does_not):
        sum_probs=0
        for verb in list_verbs:
            mlm_causal_causal_verb = verb
            token_probs,sequence=self.create_prob_dict_two_strings(mlm_first_query_token,mlm_causal_causal_verb,mlm_second_query_token,does_not)
            sum_probs+=token_probs[mlm_token_to_check]
            self.logger.debug(f"For the sentence {sequence} probability of the word {mlm_token_to_check} to occur at the end is {token_probs[mlm_token_to_check]}")
        avg=sum_probs/len(list_verbs)
        return avg

    def find_prob_for_each_query(self,verb, input_tokens):
            mlm_first_query_token=input_tokens[0]
            mlm_causal_causal_verb = verb
            mlm_second_query_token = input_tokens[1]
            sequence = str(mlm_first_query_token) + " " + str(
                mlm_causal_causal_verb) + " " + self.tokenizer.mask_token
            token_probs,sequence=self.create_prob_dict(sequence)
            prob=token_probs[mlm_second_query_token]
            self.logger.debug(f"For the sentence {sequence} probability of the word {mlm_second_query_token} to occur at the end is {token_probs[mlm_second_query_token]}")
            return prob

    def find_average_causal_mlm(self,list_verbs, input_tokens):
        sum_probs=0
        for verb in list_verbs:
            prob=self.find_prob_for_each_query(verb,input_tokens)
            sum_probs=sum_probs+prob
        avg=sum_probs/len(list_verbs)
        return avg

    def find_prob_multiple_tokens_per_verb(self,verb,flag_multi_word_token_goes_first,left_token,all_tokens_in_between,query_token):
        mlm_causal_causal_verb = verb
        if (flag_multi_word_token_goes_first):
            sequence = str(left_token) + " " + all_tokens_in_between + " " + str(
                mlm_causal_causal_verb) + " " + self.tokenizer.mask_token
        else:
            sequence = str(left_token) + " " + str(
                mlm_causal_causal_verb) + " " + all_tokens_in_between + " " + self.tokenizer.mask_token
        token_probs, sequence = self.create_prob_dict(sequence)
        prob= token_probs[query_token]
        self.logger.debug(
            f"For the sentence {sequence} probability of the word {query_token} to occur at the end is {token_probs[query_token]}")
        return prob


    def find_average_causal_mlm_multiple_tokens(self, list_verbs, left_token, all_tokens_in_between, query_token,flag_multi_word_token_goes_first):
        sum_probs=0
        for verb in list_verbs:
            prob=self.find_prob_multiple_tokens_per_verb( verb, flag_multi_word_token_goes_first, left_token,
                                               all_tokens_in_between, query_token)
            sum_probs+=prob
        avg=sum_probs/len(list_verbs)

        return avg

    def find_average_causal_mlm_with_negation(self,list_verbs, input_tokens):
        sum_probs=0
        for verb in list_verbs:
            mlm_first_query_token=input_tokens[0]
            mlm_causal_causal_verb = verb


            mlm_second_query_token = input_tokens[1]
            sequence=str(mlm_first_query_token)+" "+str(mlm_causal_causal_verb)+" "+self.tokenizer.mask_token
#            sequence=str(mlm_first_query_token)+" does not "+str(mlm_causal_causal_verb)+" "+self.tokenizer.mask_token

            input = self.tokenizer.encode(sequence, return_tensors="pt")
            mask_token_index = torch.where(input == self.tokenizer.mask_token_id)[1]
            token_logits = self.model(input).logits
            mask_token_logits = token_logits[0, mask_token_index, :]

            mask_token_logits=F.softmax(mask_token_logits)
            all_token_probs = torch.topk(mask_token_logits, len(self.tokenizer.get_vocab()), dim=1)
            alltoken_indices=all_token_probs[1]
            alltoken_probs=all_token_probs[0]
            token_probs={}
            for token_index,prob in zip(alltoken_indices[0],all_token_probs[0][0]):
                token=self.tokenizer.decode(token_index)
                token_probs[token]=prob
            sum_probs+=token_probs[mlm_second_query_token]
            self.logger.debug(f"For the sentence {sequence} probability of the word {mlm_second_query_token} to occur at the end is {token_probs[mlm_second_query_token]}")
        avg=sum_probs/len(list_verbs)
        return avg


    def rice_production_average_per_query_type(self,obj_direction_validation,list_verb_queries):
        input_tokens = ["income", "rice"]
        avg_rice = obj_direction_validation.find_average_causal_mlm(list_verb_queries, input_tokens)
        self.logger.debug(f"For INCOME IMPROVES [MASK] avg probabilityavg probability for all PROMOTES queries is {avg_rice}")

        avg_rice_production = obj_direction_validation.find_average_causal_mlm_given_input_tokens_as_string(
            list_verb_queries, "income", "rice", "production","")
        self.logger.debug(f"For INCOME IMPROVES RICE [MASK] avg probability  for all PROMOTES queries is {avg_rice_production}")

        avg_rice_production_income = (avg_rice + avg_rice_production) / 2
        self.logger.debug(f"For INCOME IMPROVES RICE PRODUCTION combined avg probability for all PROMOTES queries is {avg_rice_production_income}")

        return avg_rice_production_income


    def rice_production_average_per_query_type_with_negation(self,obj_direction_validation,list_verb_queries):
        input_tokens = ["income", "rice"]
        avg_rice = obj_direction_validation.find_average_causal_mlm_with_negation(list_verb_queries, input_tokens)
        self.logger.debug(f"For INCOME IMPROVES [MASK] avg probabilityavg probability for all PROMOTES queries is {avg_rice}")

        avg_rice_production = obj_direction_validation.find_average_causal_mlm_given_input_tokens_as_string(
            list_verb_queries, "income", "rice", "production","does not")
        self.logger.debug(f"For INCOME IMPROVES RICE [MASK] avg probability  for all PROMOTES queries is {avg_rice_production}")

        avg_rice_production_income = (avg_rice + avg_rice_production) / 2
        self.logger.debug(f"For INCOME IMPROVES RICE PRODUCTION combined avg probability for all PROMOTES queries is {avg_rice_production_income}")

        return avg_rice_production_income


    def all_single_worded_queries(self, input_tokens):
        all_averages = []

        #mirror images of each other. dictionary maping probability to verb and verb to probability
        prob_verb= {}
        verb_prob = {}

        avg = self.find_average_causal_mlm(all_promote_verbs, input_tokens)
        all_averages.append(avg.item())
        prob_verb[avg.item()] = "PROMOTES"
        verb_prob["PROMOTES"]= avg.item()
        self.logger.info(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all PROMOTES queries is {avg}")

        avg = self.find_average_causal_mlm(all_inhibits_verbs, input_tokens)
        all_averages.append(avg.item())
        prob_verb[avg.item()] = "INHIBITS"
        verb_prob["INHIBITS"] = avg.item()

        self.logger.info(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all INHIBITS queries is {avg}")

        avg = self.find_average_causal_mlm(all_causal_verbs, input_tokens)
        self.logger.info(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all CAUSAL queries is {avg}")
        all_averages.append(avg.item())
        all_averages.sort(reverse=True)
        prob_verb[avg.item()] = "NON_POLARIZED"
        verb_prob["NON_POLARIZED"] = avg.item()


        combined_sorted_string_verbs = []
        for a in all_averages:
            pera = str(prob_verb[a]) + " > "
            combined_sorted_string_verbs.append(pera)
        self.logger.info("".join(combined_sorted_string_verbs))

        prob_verb_doesnot = {}
        all_averages = []
        avg = self.find_average_causal_mlm_with_negation(all_does_not_promote_verbs,
                                                         input_tokens)
        self.logger.info(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all DOES NOT PROMOTE queries is {avg}")
        all_averages.append(avg.item())
        prob_verb_doesnot[avg.item()] = "DOES NOT PROMOTE"
        verb_prob["DOES_NOT_PROMOTE"] = avg.item()


        avg = self.find_average_causal_mlm_with_negation(all_does_not_inhibits_verbs,
                                                         input_tokens)
        self.logger.info(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all DOES NOT INHIBIT queries is {avg}")
        all_averages.append(avg.item())
        prob_verb_doesnot[avg.item()] = "DOES NOT INHIBIT"
        verb_prob["DOES_NOT_INHIBIT"] = avg.item()

        avg = self.find_average_causal_mlm_with_negation(all_does_not_cauase_verbs,
                                                         input_tokens)
        self.logger.info(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all DOES NOT CAUSE queries is {avg}")
        all_averages.append(avg.item())
        prob_verb_doesnot[avg.item()] = "DOES NOT NON_POLARIZED"
        verb_prob["DOES_NOT_NON_POLARIZED"] = avg.item()

        all_averages.sort(reverse=True)

        prob_verb_doesnot[avg.item()] = "NON_POLARIZED"
        combined_sorted_string_verbs = []
        for a in all_averages:
            pera = str(prob_verb_doesnot[a]) + " > "
            combined_sorted_string_verbs.append(pera)
        self.logger.info("".join(combined_sorted_string_verbs))

        return verb_prob


    def give_verb_types_return_multi_word_query_averages(self, split_multi_word_token, flag_multi_word_token_goes_first, partner_token, list_type_of_verbs):
        list_all_avg_probabilities = []
        word_buildup = []
        prob_of_each_sub_token_to_appear_at_end = 0
        for index, each_subtoken_in_multi_word_token in enumerate(split_multi_word_token):
            self.logger.debug("*******")
            if index == 0:
                if (flag_multi_word_token_goes_first) == True:
                    new_input_tokens = [each_subtoken_in_multi_word_token,
                                        partner_token]  # e.g,:find probability for the token 'rice' to fill 'income promotes [MASK] '
                else:
                    new_input_tokens = [partner_token, each_subtoken_in_multi_word_token]
                prob_of_each_sub_token_to_appear_at_end = self.find_average_causal_mlm(
                    list_type_of_verbs, new_input_tokens)

                self.logger.debug("--")
                self.logger.debug(
                    f"In the multiword query sub sentence: {each_subtoken_in_multi_word_token} :the average "
                    f"probability of the word {partner_token} to occur at the end with all the given verb type is {prob_of_each_sub_token_to_appear_at_end}")
                self.logger.debug("--")


            if index > 0:
                # for subsequent indicies, after zero, you have to keep building up the sentence.
                # e.g now you have to find the probability of 'production' to fill 'income promotes rice [mask]'
                first_part_of_multi_word_token=" ".join(word_buildup)
                if (flag_multi_word_token_goes_first) == True:
                    prob_of_each_sub_token_to_appear_at_end = self.find_average_causal_mlm_multiple_tokens(
                        list_type_of_verbs, first_part_of_multi_word_token, each_subtoken_in_multi_word_token, partner_token,
                        flag_multi_word_token_goes_first)

                else:
                    #find_average_causal_mlm_multiple_tokens(self, list_verbs, left_token, all_tokens_in_between, query_token,flag_multi_word_token_goes_first):
                    prob_of_each_sub_token_to_appear_at_end = self.find_average_causal_mlm_multiple_tokens(
                    list_type_of_verbs, partner_token,first_part_of_multi_word_token, each_subtoken_in_multi_word_token,
                    flag_multi_word_token_goes_first)
                self.logger.debug("--")
                self.logger.debug(
                    f"In the multiword query sub sentence: {first_part_of_multi_word_token} {each_subtoken_in_multi_word_token}:the average "
                    f"probability of the word {partner_token} to occur at the end with all the given verb type  is {prob_of_each_sub_token_to_appear_at_end}")
                self.logger.debug("--")

            word_buildup.append(each_subtoken_in_multi_word_token)
            assert prob_of_each_sub_token_to_appear_at_end > 0
            list_all_avg_probabilities.append(prob_of_each_sub_token_to_appear_at_end.item())

        all_avg_probabilities = sum(list_all_avg_probabilities) / len(
            list_all_avg_probabilities)

        return all_avg_probabilities

    def get_avg_of_multi_worded_queries(self, index, len_input_tokens,input_tokens,split_multi_word_token):
        # at this point, either the first token in input_tokens or the second token can be the multiword_one. find which one
        if index == 0:
            flag_multi_word_token_goes_first = True
        else:
            flag_multi_word_token_goes_first=False



        # pick the other token...assumption here is that one of the tokens is multiworded and the other one will be singleworded
        # .e:g.,if "rice production" is one token, the other token is "income"
        index_of_other_part_token = len_input_tokens - 1 - index
        partner_token = input_tokens[index_of_other_part_token]

        ######## all promotes relate verbs
        # for each sub token, find the probability of query token cumulatively.
        # e.g: find probability of income to fill 'rice improves [mask]'
        # then find probability of income to fill 'rice production improves [mask]'
        avg_prob_of_all_promote_queries=self.give_verb_types_return_multi_word_query_averages(split_multi_word_token, flag_multi_word_token_goes_first,
                                                                                                  partner_token, all_promote_verbs)


        self.logger.info(
            f"In the multiword query sentence {split_multi_word_token} the average "
            f"probability of the word {partner_token} to occur at the end with all_promote_verbs is {avg_prob_of_all_promote_queries}")



        ########all inhibits verbs
        all_avg_probabilities_inhibits= self.give_verb_types_return_multi_word_query_averages(split_multi_word_token,
                                                                                                    flag_multi_word_token_goes_first,
                                                                                                    partner_token,
                                                                                                    all_inhibits_verbs)
        self.logger.info(
            f"In the multiword query sentence {split_multi_word_token} the average "
            f"probability of the word {partner_token} to occur at the end with all_inhibits_verbs is {all_avg_probabilities_inhibits}")

        ########all does not promote verbs
        all_avg_probabilities_dnpromotes = self.give_verb_types_return_multi_word_query_averages(split_multi_word_token,
                                                                                                 flag_multi_word_token_goes_first,
                                                                                                 partner_token,
                                                                                                 all_does_not_promote_verbs)
        self.logger.info(
            f"In the multiword query sentence {split_multi_word_token} the average "
            f"probability of the word {partner_token} to occur at the end with all_does_not_promote_verbs is {all_avg_probabilities_dnpromotes}")

        ########all does not promote verbs
        all_avg_probabilities_dninhibits = self.give_verb_types_return_multi_word_query_averages(split_multi_word_token,
                                                                                                 flag_multi_word_token_goes_first,
                                                                                                 partner_token,
                                                                                                 all_does_not_inhibits_verbs)
        self.logger.info(
            f"In the multiword query sentence {split_multi_word_token} the average "
            f"probability of the word {partner_token} to occur at the end with all_does_not_inhibits_verbs is {all_avg_probabilities_dninhibits}")


        dict_all_avg_probs={
            "PROMOTES":avg_prob_of_all_promote_queries,
            "INHIBITS": all_avg_probabilities_inhibits,
            "DOES_NOT_PROMOTE": all_avg_probabilities_dnpromotes,
            "DOES_NOT_INHIBIT": all_avg_probabilities_dninhibits
        }

        return dict_all_avg_probs

    def find_highest_prob_between_verb_donot_verb(self, dict_verb_prob_a2b, key1, key2, overall_highest_accuracies_relations, direction_string):
        prob_key1=dict_verb_prob_a2b[key1]
        prob_key2=dict_verb_prob_a2b[key2]
        if (prob_key1 > prob_key2):
            who_is_bigger=f"{key1}>{key2}"
            self.logger.info(who_is_bigger)
            overall_highest_accuracies_relations[prob_key1]=(who_is_bigger,direction_string)

def get_data(data_file_path):
    lines=read_file(data_file_path)
    for each_row in lines:
        each_row_split=each_row.rstrip().split("|")
        list_all_causal_variables=each_row_split[0]
        list_all_effect_variables = each_row_split[1]
        return list_all_causal_variables.split(","),list_all_effect_variables.split(",")

def get_input_tokens_find_probability(model_name):
    list_all_causal_variables, list_all_effect_variables = get_data(VARIABLES_FILE)
    for causal_variable in list_all_causal_variables:
        for effect_variable in list_all_effect_variables:
            input_tokens = [causal_variable, effect_variable]
            find_causal_probability_for_tokens(model_name, input_tokens)

def convert_probability_dict_to_pretty_print(data,input_tokens,pretty_dict):
    for k, v in data.items():
        if not "polarized" in k.lower(): #the dictionary at this point has two more extra lines for NON_POLARIZED verbs. e.g impacts. dont write it to disk. this is a future work.
            pretty_key = f"{input_tokens[0]} {k} {input_tokens[1]}"
            pretty_dict[pretty_key] = v
    return pretty_dict


def find_causal_probability_for_tokens(model_name,input_tokens):
        '''Finds the probability of each effect token to appear at the end of causal token with verb.
        For example if input_tokens=[education, income], find probability of income to occur at the end of :
        education improves ______ etc.

        Note that if either of the tokens are multiworded tokens (e.g.; rice production) probabilities are recursively
        averaged. For example, for input tokens=["income", "rice production"] overall probability of rice production to
        occur at the end of income based sentences are:average(prob(income promotes rice [MASK]), prob(income promotes [MASK])'''
        obj_direction_validation = DirectionValidation(model_name,input_tokens)
        verb_prob_a2b=None
        verb_prob_b2a = None

        len_input_tokens=len(input_tokens)
        obj_direction_validation.logger.info(
            f"######going to run queries in the direction  {input_tokens[0]} to {input_tokens[1]}######")

        #There is an assumption here that one of the tokens is multiword and other wont be
        flag_found_multi_word_token=False
        for index,each_token in enumerate(input_tokens):
            split_multi_word_token=each_token.split(" ")
            if (len(split_multi_word_token) > 1):
                verb_prob_a2b= obj_direction_validation.get_avg_of_multi_worded_queries(index, len_input_tokens, input_tokens, split_multi_word_token)
                obj_direction_validation.logger.debug("end of multi token expts")
                flag_found_multi_word_token=True
        if not (flag_found_multi_word_token):
                #####for queries where both are singled worded tokens. eg: stability improves income
                # verb_prob_a2b stores the type of query as key and the average of all sub queries as its average e.g.;["DOES_NOT_NON_POLARIZED"] = avg.item()
                verb_prob_a2b = obj_direction_validation.all_single_worded_queries(input_tokens)

        ############## opposite direction

        input_tokens_reverse = [input_tokens[1], input_tokens[0]]
        obj_direction_validation.logger.info(
            f"######going to run in the opposite direction i.e {input_tokens_reverse[0]} to {input_tokens_reverse[1]}######")
        flag_found_multi_word_token = False
        for index, each_token in enumerate(input_tokens_reverse):
            split_multi_word_token = each_token.split(" ")
            if (len(split_multi_word_token) > 1):
                verb_prob_b2a = obj_direction_validation.get_avg_of_multi_worded_queries(index, len_input_tokens,
                                                                                         input_tokens_reverse,
                                                                                           split_multi_word_token)
                obj_direction_validation.logger.debug("end of multi token expts")
                flag_found_multi_word_token = True
        if not (flag_found_multi_word_token):
            #####for queries where both are singled worded tokens. eg: stability improves income
            # verb_prob_a2b stores the type of query as key and the average of all sub queries as its average e.g.;["DOES_NOT_NON_POLARIZED"] = avg.item()
            verb_prob_b2a = obj_direction_validation.all_single_worded_queries(input_tokens_reverse)

        ########find overall_highest_accuracies_relations
        # i.e by now we would have got average probabilites of all verbs in both directions. now find which one is
        # bigger. Example is PROMOTES>INHIBITS for the given set of tokens and verbs
        assert verb_prob_a2b is not None
        assert verb_prob_b2a is not None
        assert len(verb_prob_a2b.keys()) == len(verb_prob_b2a.keys())
        pretty_dict={}
        convert_probability_dict_to_pretty_print(verb_prob_a2b,input_tokens,pretty_dict)
        convert_probability_dict_to_pretty_print(verb_prob_b2a, input_tokens_reverse, pretty_dict)
        write_dict_to_json(pretty_dict,OUTPUT_FILE)

        overall_highest_accuracies_relations = {}
        obj_direction_validation.logger.info(f"Extended Summary:")
        direction_string = f"From {input_tokens[0]} to {input_tokens[1]:}"
        obj_direction_validation.logger.info(direction_string)
        obj_direction_validation.find_highest_prob_between_verb_donot_verb(verb_prob_a2b, "PROMOTES",
                                                                               "DOES_NOT_PROMOTE",
                                                                           overall_highest_accuracies_relations,
                                                                           direction_string)
        obj_direction_validation.find_highest_prob_between_verb_donot_verb(verb_prob_a2b, "INHIBITS",
                                                                               "DOES_NOT_INHIBIT",
                                                                           overall_highest_accuracies_relations,
                                                                           direction_string)

        direction_string_reverse = f"From {input_tokens[1]} to {input_tokens[0]:}"
        obj_direction_validation.logger.info(direction_string_reverse)
        obj_direction_validation.find_highest_prob_between_verb_donot_verb(verb_prob_b2a, "PROMOTES",
                                                                               "DOES_NOT_PROMOTE",
                                                                           overall_highest_accuracies_relations,
                                                                           direction_string_reverse)
        obj_direction_validation.find_highest_prob_between_verb_donot_verb(verb_prob_b2a, "INHIBITS",
                                                                               "DOES_NOT_INHIBIT",
                                                                           overall_highest_accuracies_relations,
                                                                           direction_string_reverse)
        obj_direction_validation.logger.info(f"Brief Summary:\n Overall_best=")

        # find the higheest value and print it as the best overall

        all_probs = []
        for kv in overall_highest_accuracies_relations.keys():
            all_probs.append(kv)

        if(len(all_probs)>0):
            obj_direction_validation.logger.info(overall_highest_accuracies_relations[max(all_probs)][1])
            obj_direction_validation.logger.info(overall_highest_accuracies_relations[max(all_probs)][0])

def main():
    for each_model in LIST_MODEL_NAME:
        get_input_tokens_find_probability(each_model)


if __name__ == "__main__":
    main()