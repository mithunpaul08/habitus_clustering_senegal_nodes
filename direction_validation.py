from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from glove_read_get_embed import read_eidos_stopwords
import sys
import nltk
import torch.nn.functional as F
from nltk.corpus import stopwords
from transformers import pipeline
from pprint import pprint
from utils import read_file_python_way
import torch
from data.verbs import *

import logging

class DirectionValidation:

    def __init__(self,model_name):

        self.logger = logging.getLogger(__name__)
        log_file_name = "log_directionality.log"
        FORMAT='%(message)s'
        logging.basicConfig(
            format=FORMAT,
            level=logging.INFO,
            filename=log_file_name,
            filemode='w'
        )
        self.logger.info(f"modelname={model_name}")

        self.MLM_MODEL=model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.MLM_MODEL)
        self.model = AutoModelForMaskedLM.from_pretrained(self.MLM_MODEL)

    def create_prob_dict(self,mlm_first_query_token,mlm_causal_causal_adverb):
        sequence = str(mlm_first_query_token) + " " + str(mlm_causal_causal_adverb) + " " + self.tokenizer.mask_token
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
        self.logger.debug(f"prob_dict['income']={token_probs['income']}")
        return token_probs,sequence



    def create_prob_dict_two_strings(self,mlm_first_query_token,mlm_causal_causal_adverb,mlm_second_query_token,does_not=""):
        sequence = str(mlm_first_query_token) +" "+ does_not+" " + str(mlm_causal_causal_adverb) + " "+ str(mlm_second_query_token)+" " + self.tokenizer.mask_token
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

    def find_average_causal_mlm_given_input_tokens_as_string(self,list_adverbs, mlm_first_query_token,mlm_second_query_token,mlm_token_to_check,does_not):
        sum_probs=0
        for adverb in list_adverbs:
            mlm_causal_causal_adverb = adverb
            token_probs,sequence=self.create_prob_dict_two_strings(mlm_first_query_token,mlm_causal_causal_adverb,mlm_second_query_token,does_not)
            sum_probs+=token_probs[mlm_token_to_check]
            self.logger.debug(f"For the sentence {sequence} probability of the word {mlm_token_to_check} to occur at the end is {token_probs[mlm_token_to_check]}")
        avg=sum_probs/len(list_adverbs)
        return avg

    def find_average_causal_mlm(self,list_adverbs, input_tokens):
        sum_probs=0
        for adverb in list_adverbs:
            mlm_first_query_token=input_tokens[0]
            mlm_causal_causal_adverb = adverb
            mlm_second_query_token = input_tokens[1]
            token_probs,sequence=self.create_prob_dict(mlm_first_query_token,mlm_causal_causal_adverb)
            sum_probs+=token_probs[mlm_second_query_token]
            self.logger.debug(f"For the sentence {sequence} probability of the word {mlm_second_query_token} to occur at the end is {token_probs[mlm_second_query_token]}")
        avg=sum_probs/len(list_adverbs)
        return avg


    def find_average_causal_mlm_with_negation(self,list_adverbs, input_tokens):
        sum_probs=0
        for adverb in list_adverbs:
            mlm_first_query_token=input_tokens[0]
            mlm_causal_causal_adverb = adverb


            mlm_second_query_token = input_tokens[1]
            sequence=str(mlm_first_query_token)+" does not "+str(mlm_causal_causal_adverb)+" "+self.tokenizer.mask_token
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
        avg=sum_probs/len(list_adverbs)
        return avg


    def rice_production_average_per_query_type(self,obj_direction_validation,list_adverb_queries):
        input_tokens = ["income", "rice"]
        avg_rice = obj_direction_validation.find_average_causal_mlm(list_adverb_queries, input_tokens)
        self.logger.debug(f"For INCOME IMPROVES [MASK] avg probabilityavg probability for all PROMOTES queries is {avg_rice}")

        avg_rice_production = obj_direction_validation.find_average_causal_mlm_given_input_tokens_as_string(
            list_adverb_queries, "income", "rice", "production","")
        self.logger.debug(f"For INCOME IMPROVES RICE [MASK] avg probability  for all PROMOTES queries is {avg_rice_production}")

        avg_rice_production_income = (avg_rice + avg_rice_production) / 2
        self.logger.debug(f"For INCOME IMPROVES RICE PRODUCTION combined avg probability for all PROMOTES queries is {avg_rice_production_income}")

        return avg_rice_production_income


    def rice_production_average_per_query_type_with_negation(self,obj_direction_validation,list_adverb_queries):
        input_tokens = ["income", "rice"]
        avg_rice = obj_direction_validation.find_average_causal_mlm_with_negation(list_adverb_queries, input_tokens)
        self.logger.debug(f"For INCOME IMPROVES [MASK] avg probabilityavg probability for all PROMOTES queries is {avg_rice}")

        avg_rice_production = obj_direction_validation.find_average_causal_mlm_given_input_tokens_as_string(
            list_adverb_queries, "income", "rice", "production","does not")
        self.logger.debug(f"For INCOME IMPROVES RICE [MASK] avg probability  for all PROMOTES queries is {avg_rice_production}")

        avg_rice_production_income = (avg_rice + avg_rice_production) / 2
        self.logger.debug(f"For INCOME IMPROVES RICE PRODUCTION combined avg probability for all PROMOTES queries is {avg_rice_production_income}")

        return avg_rice_production_income


    def all_queries(self,input_tokens):
        all_averages = []
        prob_adverb= {}
        adverb_prob = {}

        avg = self.find_average_causal_mlm(all_promote_adverbs, input_tokens)
        all_averages.append(avg.item())
        prob_adverb[avg.item()] = "PROMOTES"
        adverb_prob["PROMOTES"]= avg.item()
        self.logger.info(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all PROMOTES queries is {avg}")

        avg = self.find_average_causal_mlm(all_inhibits_adverbs, input_tokens)
        all_averages.append(avg.item())
        prob_adverb[avg.item()] = "INHIBITS"
        adverb_prob["INHIBITS"] = avg.item()

        self.logger.info(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all INHIBITS queries is {avg}")

        avg = self.find_average_causal_mlm(all_causal_adverbs, input_tokens)
        self.logger.info(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all CAUSAL queries is {avg}")
        all_averages.append(avg.item())
        all_averages.sort(reverse=True)
        prob_adverb[avg.item()] = "NON_POLARIZED"
        adverb_prob["NON_POLARIZED"] = avg.item()


        combined_sorted_string_adverbs = []
        for a in all_averages:
            pera = str(prob_adverb[a]) + " > "
            combined_sorted_string_adverbs.append(pera)
        self.logger.info("".join(combined_sorted_string_adverbs))

        prob_adverb_doesnot = {}
        all_averages = []
        avg = self.find_average_causal_mlm_with_negation(all_promote_adverbs_for_negation,
                                                                             input_tokens)
        self.logger.info(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all DOES NOT PROMOTE queries is {avg}")
        all_averages.append(avg.item())
        prob_adverb_doesnot[avg.item()] = "DOES NOT PROMOTE"
        adverb_prob["DOES_NOT_PROMOTE"] = avg.item()


        avg = self.find_average_causal_mlm_with_negation(all_inhibits_adverbs_for_negation,
                                                                             input_tokens)
        self.logger.info(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all DOES NOT INHIBIT queries is {avg}")
        all_averages.append(avg.item())
        prob_adverb_doesnot[avg.item()] = "DOES NOT INHIBIT"
        adverb_prob["DOES_NOT_INHIBIT"] = avg.item()

        avg = self.find_average_causal_mlm_with_negation(all_causal_adverbs_for_negation,
                                                                             input_tokens)
        self.logger.info(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all DOES NOT CAUSE queries is {avg}")
        all_averages.append(avg.item())
        prob_adverb_doesnot[avg.item()] = "DOES NOT NON_POLARIZED"
        adverb_prob["DOES_NOT_NON_POLARIZED"] = avg.item()

        all_averages.sort(reverse=True)

        prob_adverb_doesnot[avg.item()] = "NON_POLARIZED"
        combined_sorted_string_adverbs = []
        for a in all_averages:
            pera = str(prob_adverb_doesnot[a]) + " > "
            combined_sorted_string_adverbs.append(pera)
        self.logger.info("".join(combined_sorted_string_adverbs))

        return adverb_prob

    def find_highest_prob_between_adverb_donot_adverb(self,dict_adverb_prob_a2b,key1, key2,overall_highest_accuracies_relations):
        prob_key1=dict_adverb_prob_a2b[key1]
        prob_key2=dict_adverb_prob_a2b[key2]
        if (prob_key1 > prob_key2):
            print(f"{key1}>{key2}")
            overall_highest_accuracies_relations[]


def main():

    obj_direction_validation = DirectionValidation("distilbert-base-uncased")

    input_tokens = ["education", "stability"]
    adverb_prob_a2b=obj_direction_validation.all_queries(input_tokens)


    overall_highest_accuracies_relations={}

    print(f"Extended Summary:")
    print(f"From {input_tokens[0]} to {input_tokens[1]}")
    obj_direction_validation.find_highest_prob_between_adverb_donot_adverb(adverb_prob_a2b,"PROMOTES","DOES_NOT_PROMOTE",overall_highest_accuracies_relations)
    obj_direction_validation.find_highest_prob_between_adverb_donot_adverb(adverb_prob_a2b, "INHIBITS",
                                                                           "DOES_NOT_INHIBIT",overall_highest_accuracies_relations)

    input_tokens = ["stability", "education"]
    adverb_prob_b2a = obj_direction_validation.all_queries(input_tokens)
    assert len(adverb_prob_a2b.keys()) == len(adverb_prob_b2a.keys())

    print(f"From {input_tokens[0]} to {input_tokens[1]}")
    obj_direction_validation.find_highest_prob_between_adverb_donot_adverb(adverb_prob_b2a, "PROMOTES",
                                                                           "DOES_NOT_PROMOTE",overall_highest_accuracies_relations)
    obj_direction_validation.find_highest_prob_between_adverb_donot_adverb(adverb_prob_b2a, "INHIBITS",
                                                                           "DOES_NOT_INHIBIT",overall_highest_accuracies_relations)

    #########this is being hardcodeed for the reverse direction "rice production" todo:should make this general- i.e even if there are multiple tokens in one token, do recursively aveerage
#     prob_adverb = {}
#     all_averages = []
#     input_tokens = [ "income","rice production"]
#     avg=obj_direction_validation.rice_production_average_per_query_type(obj_direction_validation,all_promote_adverbs)
#     all_averages.append(avg.item())
#     prob_adverb[avg.item()]="PROMOTES"
#     print(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all PROMOTES queries is {avg}")
#
#
#     avg=obj_direction_validation.rice_production_average_per_query_type(obj_direction_validation,all_inhibits_adverbs)
#     all_averages.append(avg.item())
#     prob_adverb[avg.item()] = "INHIBITS"
#     print(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all INHIBITS queries is {avg}")
#
#
#     avg=obj_direction_validation.rice_production_average_per_query_type(obj_direction_validation,all_causal_adverbs)
#     print(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all CAUSAL queries is {avg}")
#     all_averages.append(avg.item())
#     all_averages.sort(reverse=True)
#     prob_adverb[avg.item()] = "NON_POLARIZED"
#     combined_sorted_string_adverbs=[]
#     for a in all_averages:
#         pera=str(prob_adverb[a])+" > "
#         combined_sorted_string_adverbs.append(pera)
#     print("".join(combined_sorted_string_adverbs))
#
# ####income and rice production with negation
#     prob_adverb = {}
#     all_averages = []
#     input_tokens = ["income", "rice production"]
#     avg = obj_direction_validation.rice_production_average_per_query_type_with_negation(obj_direction_validation, all_promote_adverbs_for_negation)
#     all_averages.append(avg.item())
#     prob_adverb[avg.item()] = "DOES NOT PROMOTE"
#     print(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all DOES NOT PROMOTES queries is {avg}")
#
#     avg = obj_direction_validation.rice_production_average_per_query_type_with_negation(obj_direction_validation,
#                                                                           all_inhibits_adverbs_for_negation)
#     all_averages.append(avg.item())
#     prob_adverb[avg.item()] = "DOES NOT INHIBIT"
#     print(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all DOES NOT INHIBITS queries is {avg}")
#
#     avg = obj_direction_validation.rice_production_average_per_query_type_with_negation(obj_direction_validation, all_causal_adverbs_for_negation)
#     print(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all DOES NOT CAUSAL queries is {avg}")
#     all_averages.append(avg.item())
#     all_averages.sort(reverse=True)
#     prob_adverb[avg.item()] = "DOES NOT NON_POLARIZED"
#     combined_sorted_string_adverbs = []
#     for a in all_averages:
#         pera = str(prob_adverb[a]) + " > "
#         combined_sorted_string_adverbs.append(pera)
#     print("".join(combined_sorted_string_adverbs))


if __name__ == "__main__":
    main()