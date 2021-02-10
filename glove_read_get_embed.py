import sys
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy import random
import pandas as pd


GLOVE_PATH_CLARA_SERVER="/data/nlp/corpora/glove/glove.840B.300d.10f.txt"
GLOVE_PATH_LOCAL="./glove_bottom10.txt"

random.seed(3)

words_embeddings={}



def read_from_glove():
    lines=open(GLOVE_PATH_LOCAL,mode='r')
    for line in lines:
            all_words=line.split()
            word=all_words[0]
            emb=np.array(all_words[1:len(all_words)])
            emb=emb.astype(np.float)
            embdash=np.reshape(emb, [1, -1])
            words_embeddings[word]=embdash

def read_from_glove_assimilate_using_pandas():
    df = pd.read_csv(GLOVE_PATH_CLARA_SERVER,skiprows=[0],header=None)
    #df = pd.read_csv(GLOVE_PATH_LOCAL)
    for index,line in df.iterrows():
            all_words=line[0].split()
            word=all_words[0]
            emb=np.array(all_words[1:len(all_words)])
            emb=emb.astype(np.float)
            embdash=np.reshape(emb, [1, -1])
            words_embeddings[word]=embdash

#read_from_glove_assimilate_using_pandas()
read_from_glove()
def get_embedding_given_token(tk):
    assert words_embeddings is not None
    assert len(words_embeddings.keys()) > 0
    return words_embeddings.get(tk,None)



def return_pairwise(word1,word2):
    read_from_glove_assimilate_using_pandas()
    emb_word1=get_embedding_given_token(word1)
    emb_word2 = get_embedding_given_token(word2)
    return cosine_similarity(emb_word1, emb_word2)


#
# cos=return_pairwise('zulchzulu','wried')
# print(cos)
