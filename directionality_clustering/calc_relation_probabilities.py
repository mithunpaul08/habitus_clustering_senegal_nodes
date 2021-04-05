import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch.nn.functional as F
from data.verbs import *
from utils import *
import argparse

MLM_MODELS=["bert-base-uncased","distilbert-base-uncased"]

def parse_arguments():
    argparser = argparse.ArgumentParser("to parse causal documents")
    argparser.add_argument("input_file_name", help="name of the input file where causal and effect variables are kept")
    args = argparser.parse_args()
    return args.input_file_name

# prob(effect | cause)
def create_prob_dict(sequence,tokenizer,model):
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

def prob_expected_word(text_with_mask, expected_word, tokenizer, model):
    token_probs, text_with_mask = create_prob_dict(text_with_mask,tokenizer,model)
    prob= token_probs[expected_word]
    return prob

def calc_rel_prob(cause, effect, triggers,tokenizer,model):
    probabilities = []
    effect_tokens = effect.split(' ')
    for trigger in triggers:
        for i in range(len(effect_tokens)):
            effect_chunk = ' '.join(effect_tokens[:i])
            text = f'{cause} {trigger} {effect_chunk} [MASK]'
            prob_effect=prob_expected_word(text, effect_tokens[i], tokenizer, model)
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

names_triggers={
    "PROMOTES":all_promote_verbs,
    "INHIBITS":all_inhibits_verbs,
    "DOES_NOT_PROMOTE":all_does_not_promote_verbs,
    "DOES_NOT_INHIBT":all_does_not_inhibits_verbs
}

def calc_average_probabilities(input_file_name):
    data=read_data(input_file_name)

    # output file for overall across models average
    output_file_overall = f"outputs/overall_probabilities.tsv"
    initalize_file(output_file_overall)

    # for each type of trigger verb
    for trigger_group_name,triggers in names_triggers.items():


        # for each line in the input tsv file
        for id_cause, cause_synonyms in data.items():
            for id_effect, effect_synonyms in data.items():
                if not (id_cause == id_effect):
                    trigger_prob_avg_across_models = []
                    for each_model in MLM_MODELS:
                        # output file for per model average
                        output_file_model = f"outputs/{each_model}_probabilities.tsv"
                        # empty out the output file if it exists from previous runs
                        #initalize_file(output_file_model)

                        tokenizer = AutoTokenizer.from_pretrained(each_model)
                        model = AutoModelForMaskedLM.from_pretrained(each_model)

                        # probabilities for each element in cartesian product
                        probabilities = []
                        for cause in cause_synonyms:
                            for effect in effect_synonyms:
                                p = calc_rel_prob(cause, effect, triggers,tokenizer,model)
                                probabilities.append(p)
                        # calculate probability average
                        avg_prob = float(sum(probabilities)) / float(len(probabilities))
                        output=f"{id_cause}\t{id_effect}\t{trigger_group_name}\t{avg_prob}\n"
                        append_to_file(output,output_file_model)
                        trigger_prob_avg_across_models.append(avg_prob)
                    avg_prob_across_models = float(sum(trigger_prob_avg_across_models)) / float(len(trigger_prob_avg_across_models))
                    output = f"{id_cause}\t{id_effect}\t{trigger_group_name}\t{avg_prob_across_models}\n"
                    append_to_file(output, output_file_overall)

if __name__ == "__main__":
        input_file=parse_arguments()
        calc_average_probabilities(input_file)

