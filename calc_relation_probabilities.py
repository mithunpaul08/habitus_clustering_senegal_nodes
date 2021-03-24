import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch.functional as F

MLM_MODEL="distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MLM_MODEL)
model = AutoModelForMaskedLM.from_pretrained(MLM_MODEL)

# prob(effect | cause)
def create_prob_dict(sequence):
        input = self.tokenizer.encode(sequence, return_tensors="pt")
        mask_token_index = torch.where(input == self.tokenizer.mask_token_id)[1]
        token_logits = self.model(input).logits
        mask_token_logits = token_logits[0, mask_token_index, :]

        mask_token_logits = F.softmax(mask_token_logits)
        all_token_probs = torch.topk(mask_token_logits, len(self.tokenizer.get_vocab()), dim=1)
        alltoken_indices = all_token_probs[1]
        token_probs = {}
        for token_index, prob in zip(alltoken_indices[0], all_token_probs[0][0]):
            token = self.tokenizer.decode(token_index)
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
            text = f'{cause} {trigger} {effect_chunk}[MASK]'
            print(text)
            #probabilities.append(prob(text, effect_tokens[i]))
    #avg_prob = sum(probabilities) / len(probabilities)
    return 0
def read_data(filename):
    with open(filename) as f:
        for line in f:
            [lhs, rhs] = line.strip().split('\t')
            cause_synonyms = lhs.split('|')
            effect_synonyms = rhs.split('|')
            yield cause_synonyms, effect_synonyms
promotes_triggers = ['promotes']
filename="data/query_directionality_variables.csv"

def calc_average_probabilities():
    # for each line in tsv file
    for cause_synonyms, effect_synonyms in read_data(filename):
        # probabilities for each element in cartesian product
        probabilities = []
        for cause in cause_synonyms:
            for effect in effect_synonyms:
                p = calc_rel_prob(cause, effect, promotes_triggers)
                probabilities.append(p)
        # calculate probability average
        avg_prob = sum(probabilities) / len(probabilities)
        # TODO save to file



if __name__ == "__main__":
    calc_average_probabilities()