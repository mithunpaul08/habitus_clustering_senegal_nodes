from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from glove_read_get_embed import read_eidos_stopwords
import sys
import nltk
import torch.nn.functional as F
from nltk.corpus import stopwords
from transformers import pipeline
from pprint import pprint




import torch
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-cased")
sequence = f"rainfall increases {tokenizer.mask_token}."
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

print(token_probs['rice'])

top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))

sys.exit()

eidos_stop_words = read_eidos_stopwords()
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model =AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
seq=f"rainfall increases {tokenizer.mask_token}"

input = tokenizer.encode(seq,return_tensors='pt')
mask_token_index=torch.where(input==tokenizer.mask_token_id)[1]
token_logits=model(input).logits

mask_token_logits = (token_logits[0, mask_token_index, :])
mask_softmax=F.softmax(mask_token_logits)

top_5_tokens = torch.topk(mask_softmax, 5, dim=1).indices[0].tolist()
for token in top_5_tokens:
    print(tokenizer.decode([token]))

#top_tokens = torch.topk(mask_token_logits, len(tokenizer.get_vocab()), dim=1).indices[0].tolist()
# for index,each_token_index in enumerate(top_tokens):
#
#     token=tokenizer.decode(each_token_index)
#     if (token not in eidos_stop_words) or (token not in stopwords.words('english')) :
#         print(token)
#     if index>20:
#         sys.exit()
