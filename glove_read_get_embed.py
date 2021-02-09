import sys
from sklearn.metrics.pairwise import cosine_similarity

words_embeddings={}
with open('glove_bottom10.txt',encoding='utf-8',mode='r') as f:
    for line in f:
        all_words=line.split()
        word=all_words[0]
        print(word)
        emb=all_words[1:len(all_words)]
        words_embeddings[word]=emb


pairwise_similarities=cosine_similarity(words_embeddings['xalisae'], words_embeddings['wried'])
print(pairwise_similarities)