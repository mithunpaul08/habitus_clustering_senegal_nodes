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

class DirectionValidation:

    def __init__(self,model_name):
        self.MLM_MODEL=model_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.MLM_MODEL)
        self.model = AutoModelForMaskedLM.from_pretrained(self.MLM_MODEL)

    lines=read_file_python_way(filename="data/query_directionality.csv")






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
        #print(f"prob_dict['income']={token_probs['income']}")
        return token_probs,sequence

    def find_average_causal_mlm(self,list_adverbs, input_tokens):
        sum_probs=0
        for adverb in list_adverbs:
            mlm_first_query_token=input_tokens[0]
            mlm_causal_causal_adverb = adverb
            mlm_second_query_token = input_tokens[1]
            token_probs,sequence=self.create_prob_dict(mlm_first_query_token,mlm_causal_causal_adverb)
            sum_probs+=token_probs[mlm_second_query_token]
            #print(f"For the sentence {sequence} probability of the word {mlm_second_query_token} to occur at the end is {token_probs[mlm_second_query_token]}")
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
            print(f"For the sentence {sequence} probability of the word {mlm_second_query_token} to occur at the end is {token_probs[mlm_second_query_token]}")
        avg=sum_probs/len(list_adverbs)
        return avg

    #
    # def find_average_causal_mlm_two_tokens(list_adverbs, input_tokens):
    #     sum_probs=0
    #     for adverb in list_adverbs:
    #         mlm_first_query_token=input_tokens[0]
    #         #if a token in the 2nd query has more than one tokens in it (e.g:rice production), recursively go through it, until you hit one token, then calculate probability, then add it up+average it
    #         mlm_first_query_token_split=mlm_first_query_token.split( )
    #         mlm_causal_causal_adverb = adverb
    #         mlm_second_query_token = input_tokens[1]
    #         if len(mlm_first_query_token_split) >1:
    #             for each_sub_token in mlm_first_query_token_split:
    #                 find_average_causal_mlm(list_adverbs, [each_sub_token,mlm_second_query_token])
    #         else:
    #
    #             sequence=str(mlm_first_query_token)+" "+str(mlm_causal_causal_adverb)+" "+tokenizer.mask_token
    #             input = tokenizer.encode(sequence, return_tensors="pt")
    #             mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
    #             token_logits = model(input).logits
    #             mask_token_logits = token_logits[0, mask_token_index, :]
    #
    #             mask_token_logits=F.softmax(mask_token_logits)
    #             all_token_probs = torch.topk(mask_token_logits, len(tokenizer.get_vocab()), dim=1)
    #             alltoken_indices=all_token_probs[1]
    #             alltoken_probs=all_token_probs[0]
    #             token_probs={}
    #             for token_index,prob in zip(alltoken_indices[0],all_token_probs[0][0]):
    #                 token=tokenizer.decode(token_index)
    #                 token_probs[token]=prob
    #             sum_probs+=token_probs[mlm_second_query_token]
    #             print(f"For the sentence {sequence} probability of the word {mlm_second_query_token} to occur at the end is {token_probs[mlm_second_query_token]}")
    #         avg=sum_probs/len(list_adverbs)
    #         return avg


input_tokens = ["rice production", "income"]

def main():
    obj_direction_validation=DirectionValidation("distilbert-base-uncased")
    avg=obj_direction_validation.find_average_causal_mlm(all_promote_adverbs,input_tokens)
    print(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all PROMOTES queries is {avg}")
# avg=find_average_causal_mlm(all_inhibits_adverbs,input_tokens)
# print(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all INHIBITS queries is {avg}")
# avg=find_average_causal_mlm(all_causal_adverbs,input_tokens)
# print(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all CAUSAL queries is {avg}")

    # avg=find_average_causal_mlm_with_negation(all_promote_adverbs_for_negation,input_tokens)
    # print(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all DOES NOT PROMOTE queries is {avg}")
    # avg=find_average_causal_mlm_with_negation(all_inhibits_adverbs_for_negation,input_tokens)
    # print(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all DOES NOT INHIBIT queries is {avg}")
    # avg=find_average_causal_mlm_with_negation(all_causal_adverbs_for_negation,input_tokens)
    # print(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all DOES NOT CAUSE queries is {avg}")


    #this is being hardcodeed for "rice production" todo:should make this general
    # input_tokens=["rice", "income"]
    # avg_rice=find_average_causal_mlm(all_promote_adverbs, input_tokens)
    # input_tokens=["production", "income and rice"]
    # avg_production=find_average_causal_mlm(all_promote_adverbs, input_tokens)
if __name__ == "__main__":
    main()