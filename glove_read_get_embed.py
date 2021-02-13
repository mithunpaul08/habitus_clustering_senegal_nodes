import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

GLOVE_PATH_CLARA_SERVER="/data/nlp/corpora/glove/glove.840B.300d.no_proc_header.txt"
GLOVE_PATH_LOCAL="./small_glove10f.txt"
EIDOS_STOP_WORDS='./eidos_stopwords.txt'


words_embeddings={}


def read_file_python_way(filename):
    lines=open(filename,mode='r')
    return lines

def read_eidos_stopwords():
    eidos_stopwords = []
    lines = read_file_python_way(EIDOS_STOP_WORDS)
    for line in lines:
        if not line.startswith("#"):
            eidos_stopwords.append(line.rstrip())
    return eidos_stopwords

def read_from_glove():
    lines=read_file_python_way(GLOVE_PATH_LOCAL)
    for line in lines:
            all_words=line.split()
            word=all_words[0]
            emb=np.array(all_words[1:len(all_words)])
            emb=emb.astype(np.float)
            embdash=np.reshape(emb, [1, -1])
            words_embeddings[word]=embdash


read_from_glove()
def get_embedding_given_token(tk):
    assert words_embeddings is not None
    assert len(words_embeddings.keys()) > 0
    return words_embeddings.get(tk,None)



def return_pairwise_cosine_similarity(word1,word2):
    emb_word1=get_embedding_given_token(word1)
    emb_word2 = get_embedding_given_token(word2)
    return cosine_similarity(emb_word1, emb_word2)


