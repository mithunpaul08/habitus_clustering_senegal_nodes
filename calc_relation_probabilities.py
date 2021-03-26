import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch.nn.functional as F
from data.verbs import *
from utils import *

MLM_MODEL="distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MLM_MODEL)
model = AutoModelForMaskedLM.from_pretrained(MLM_MODEL)
OUTPUT_FILE="outputs/probabilities.tsv"

# prob(effect | cause)
def create_prob_dict(sequence):
        input = tokenizer.encode(sequence, return_tensors="pt")
        mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
        token_logits = model(input).logits
        mask_token_logits = token_logits[0, mask_token_index, :]
        mask_token_logits = F.softmax(mask_token_logits)
        all_token_probs = torch.topk(mask_token_logits, len(tokenizer.get_vocab()), dim=1)
        alltoken_indices = all_token_probs[1]
        token_probs = {}
        for token_index, prob in zip(alltoken_indices[0], all_token_probs[0][0]):
            token = tokenizer.decode(token_index)
            token_probs[token] = prob
        return token_probs,sequence

def prob(text_with_mask, expected_word):
    token_probs, text_with_mask = create_prob_dict(text_with_mask)
    prob= token_probs[expected_word]
    return prob

def calc_rel_prob(cause, effect, triggers):
    probabilities = []
    effect_tokens = effect.split(' ')
    for trigger in triggers:
        for i in range(len(effect_tokens)):
            effect_chunk = ' '.join(effect_tokens[:i])
            text = f'{cause} {trigger} {effect_chunk} [MASK]'
            prob_effect=prob(text, effect_tokens[i])
            probabilities.append(prob_effect)
            #print(f"{text}:{prob_effect}")
    avg_prob = float(sum(probabilities)) / float(len(probabilities))
    return avg_prob

def read_data(filename):
    id_variables={}
    with open(filename) as f:
        for line in f:
            line_split = line.strip().split('\t')
            id=line_split[0]
            cause_effect_synonyms=line_split[1:]
            id_variables[id]=cause_effect_synonyms
    return id_variables
promotes_triggers = all_promote_verbs
filename="data/query_directionality_variables.csv"

def calc_average_probabilities():
    # for each line in tsv file
    data=read_data(filename)
    for id_cause, cause_synonyms in data.items():
        for id_effect, effect_synonyms in data.items():
            if not (id_cause == id_effect):
                # probabilities for each element in cartesian product
                probabilities = []
                for cause in cause_synonyms:
                    for effect in effect_synonyms:
                        p = calc_rel_prob(cause, effect, promotes_triggers)
                        probabilities.append(p)
                # calculate probability average
                avg_prob = float(sum(probabilities)) / float(len(probabilities))
                output=f"{id_cause}\t{id_effect}\tPROMOTES\t{avg_prob}\n"
                append_to_file(output,OUTPUT_FILE)



if __name__ == "__main__":
    initalize_file(OUTPUT_FILE)
    calc_average_probabilities()

