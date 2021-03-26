import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch.nn.functional as F
from data.verbs import *
from utils import *
import argparse
MLM_MODEL="distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MLM_MODEL)
model = AutoModelForMaskedLM.from_pretrained(MLM_MODEL)
OUTPUT_FILE="outputs/probabilities.tsv"

def parse_arguments():
    argparser = argparse.ArgumentParser("to parse causal documents")
    argparser.add_argument("input_file_name", help="name of the input file where causal and effect variables are kept")
    args = argparser.parse_args()
    return args.input_file_name

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
            print(f"{text}:{prob_effect}")
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

names_triggers={
    "PROMOTES":all_promote_verbs,
    "INHIBITS":all_inhibits_verbs,
    "DOES_NOT_PROMOTE":all_does_not_promote_verbs,
    "DOES_NOT_INHIBT":all_does_not_inhibits_verbs
}


#filename="data/query_directionality_variables.csv"

def calc_average_probabilities(input_file_name):
    # for each line in tsv file
    data=read_data(input_file_name)
    for name,triggers in names_triggers.items():
        for id_cause, cause_synonyms in data.items():
            for id_effect, effect_synonyms in data.items():
                if not (id_cause == id_effect):
                    # probabilities for each element in cartesian product
                    probabilities = []
                    for cause in cause_synonyms:
                        for effect in effect_synonyms:
                            p = calc_rel_prob(cause, effect, triggers)
                            probabilities.append(p)
                    # calculate probability average
                    avg_prob = float(sum(probabilities)) / float(len(probabilities))
                    output=f"{id_cause}\t{id_effect}\t{name}\t{avg_prob}\n"
                    append_to_file(output,OUTPUT_FILE)



if __name__ == "__main__":
    input_file_name=parse_arguments()
    initalize_file(OUTPUT_FILE)
    calc_average_probabilities(input_file_name)

