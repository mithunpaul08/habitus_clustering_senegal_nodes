import sys
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

words_embeddings={}
with open('glove_bottom10.txt',encoding='utf-8',mode='r') as f:
    for line in f:
        all_words=line.split()
        word=all_words[0]
        emb=np.array(all_words[1:len(all_words)])
        embdash=np.reshape(emb, [1, -1])
        words_embeddings[word]=embdash






pairwise_similarities=cosine_similarity(words_embeddings['xalisae'], words_embeddings['wried'])
print(pairwise_similarities)