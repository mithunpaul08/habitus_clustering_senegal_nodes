import sys
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy import random
import pandas as pd

words_embeddings={}
def read_from_glove_assimilate():
    df = pd.read_csv('./glove_bottom10.txt')
    for index,line in df.iterrows():
            all_words=line[0].split()
            word=all_words[0]
            emb=np.array(all_words[1:len(all_words)])
            emb=emb.astype(np.float)
            embdash=np.reshape(emb, [1, -1])
            words_embeddings[word]=embdash

read_from_glove_assimilate()
def get_embedding_given_token(tk):
    assert words_embeddings is not None
    assert len(words_embeddings.keys()) > 0
    emb_word = words_embeddings.get(tk,None)
    flag_out_of_vocab=False
    #if the word is out of glove vocab, create a random initialization
    #todo: switch to xavier glorot instead of random.rand
    if emb_word is None:
        flag_out_of_vocab= True
        emb_word=random.rand(1,300)
    assert emb_word is not None
    return emb_word, flag_out_of_vocab



def return_pairwise(word1,word2):
    read_from_glove_assimilate()
    emb_word1=get_embedding_given_token(word1)
    emb_word2 = get_embedding_given_token(word2)
    return cosine_similarity(emb_word1, emb_word2)


#
# cos=return_pairwise('zulchzulu','wried')
# print(cos)