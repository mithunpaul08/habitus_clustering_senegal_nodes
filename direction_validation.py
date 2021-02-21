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
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-cased")

lines=read_file_python_way(filename="data/query_directionality.csv")

all_promote_adverbs=["improves","accelerates","boosts","improves","compounds","enhances","escalates","increases","facilitates","spikes"]
all_inhibits_adverbs=["diminishes","drops","drains","exhausts","impairs","inhibits","hampers","hinders","eliminates","disrupts"]
all_causal_adverbs=["influences","affects","alters","modifies","impacts","changes","displaces"]

# in negation you want DOES NOT INFLUENCE instead of DOES NOT INFLUENCES
all_promote_adverbs_for_negation=["improve","accelerate","boost","improve","compound","enhance","escalate","increase","facilitate","spike"]
all_inhibits_adverbs_for_negation=["diminish","drop","drain","exhaust","impair","inhibit","hamper","hinder","eliminate","disrupt"]
all_causal_adverbs_for_negation=["influence","affect","alter","modify","impact","change","displace"]


input_tokens=["rice production", "income"]

def find_average_causal_mlm(list_adverbs, input_tokens):
    sum_probs=0
    for adverb in list_adverbs:
        mlm_first_query_token=input_tokens[0]
        mlm_causal_causal_adverb = adverb
        mlm_second_query_token = input_tokens[1]
        sequence=str(mlm_first_query_token)+" "+str(mlm_causal_causal_adverb)+" "+tokenizer.mask_token
        input = tokenizer.encode(sequence, return_tensors="pt")
        mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
        token_logits = model(input).logits
        mask_token_logits = token_logits[0, mask_token_index, :]

        mask_token_logits=F.softmax(mask_token_logits)
        all_token_probs = torch.topk(mask_token_logits, len(tokenizer.get_vocab()), dim=1)
        alltoken_indices=all_token_probs[1]
        alltoken_probs=all_token_probs[0]
        token_probs={}
        for token_index,prob in zip(alltoken_indices[0],all_token_probs[0][0]):
            token=tokenizer.decode(token_index)
            token_probs[token]=prob
        sum_probs+=token_probs[mlm_second_query_token]
        print(f"For the sentence {sequence} probability of the word {mlm_second_query_token} to occur at the end is {token_probs[mlm_second_query_token]}")
    avg=sum_probs/len(list_adverbs)
    return avg


def find_average_causal_mlm_with_negation(list_adverbs, input_tokens):
    sum_probs=0
    for adverb in list_adverbs:
        mlm_first_query_token=input_tokens[0]
        mlm_causal_causal_adverb = adverb


        mlm_second_query_token = input_tokens[1]
        sequence=str(mlm_first_query_token)+" does not "+str(mlm_causal_causal_adverb)+" "+tokenizer.mask_token
        input = tokenizer.encode(sequence, return_tensors="pt")
        mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
        token_logits = model(input).logits
        mask_token_logits = token_logits[0, mask_token_index, :]

        mask_token_logits=F.softmax(mask_token_logits)
        all_token_probs = torch.topk(mask_token_logits, len(tokenizer.get_vocab()), dim=1)
        alltoken_indices=all_token_probs[1]
        alltoken_probs=all_token_probs[0]
        token_probs={}
        for token_index,prob in zip(alltoken_indices[0],all_token_probs[0][0]):
            token=tokenizer.decode(token_index)
            token_probs[token]=prob
        sum_probs+=token_probs[mlm_second_query_token]
        print(f"For the sentence {sequence} probability of the word {mlm_second_query_token} to occur at the end is {token_probs[mlm_second_query_token]}")
    avg=sum_probs/len(list_adverbs)
    return avg



# avg=find_average_causal_mlm(all_promote_adverbs,input_tokens)
# print(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all PROMOTES queries is {avg}")
# avg=find_average_causal_mlm(all_inhibits_adverbs,input_tokens)
# print(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all INHIBITS queries is {avg}")
# avg=find_average_causal_mlm(all_causal_adverbs,input_tokens)
# print(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all CAUSAL queries is {avg}")

avg=find_average_causal_mlm_with_negation(all_promote_adverbs_for_negation,input_tokens)
print(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all DOES NOT PROMOTE queries is {avg}")
avg=find_average_causal_mlm_with_negation(all_inhibits_adverbs_for_negation,input_tokens)
print(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all DOES NOT INHIBIT queries is {avg}")
avg=find_average_causal_mlm_with_negation(all_causal_adverbs_for_negation,input_tokens)
print(f"For {input_tokens[0]} to {input_tokens[1]}  avg probability  for all DOES NOT CAUSE queries is {avg}")
