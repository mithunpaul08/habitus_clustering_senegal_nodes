import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords


GLOVE_PATH_CLARA_SERVER="/data/nlp/corpora/glove/glove.840B.300d.10f.txt"
GLOVE_PATH_LOCAL="./small_glove10f.txt"



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



read_from_glove()
def get_embedding_given_token(tk):
    assert words_embeddings is not None
    assert len(words_embeddings.keys()) > 0
    return words_embeddings.get(tk,None)


#read causes and effects
dataset=pd.read_csv('./senegal_africa.csv')
causes = dataset.iloc[:, [2]].values

#usee this if you want effeects
effects = dataset.iloc[:, [5]].values

combined_causes_effects=[]


for cause in causes:
    combined_causes_effects.append(cause[0])
for effect in effects:
    combined_causes_effects.append(effect[0])

def split_concept_get_combined_embedding(concept_name):
    emb_total=np.full([1,300],0.00000001)
    cause_split_tokens = concept_name.split()
    total_no_of_tokens=len(cause_split_tokens)

    #for each sub token, get embedding of them, and get the average of all n token embeddings as the concepts overall embedding value
    for each_token in cause_split_tokens:
        if each_token in stopwords.words('english'):
            continue
        emb_raw=get_embedding_given_token(each_token)
        if emb_raw is None:
            continue
        else:
            emb_total+=emb_raw
    emb_total=emb_total/total_no_of_tokens #divide by number of tokens to get average embedding for this concept
    return emb_total



def get_embedding_given_token(tk):
    assert words_embeddings is not None
    assert len(words_embeddings.keys()) > 0
    return words_embeddings.get(tk,None)



def return_pairwise_cosine_similarity(word1,word2):
    emb_word1=get_embedding_given_token(word1)
    emb_word2 = get_embedding_given_token(word2)
    return cosine_similarity(emb_word1, emb_word2)



for index,(concepts) in enumerate(combined_causes_effects):
    concept_name=concepts
    all_embeddings=split_concept_get_combined_embedding(concept_name)
    if all_embeddings is not None:
        pass
        #do whatever
    else:
        print(f"found that the concept at index {index} and concept={concepts} had a zero embedding")



# use this code if you want to calculate cosinee similarity between the concepts you find above
cos1 = return_pairwise_cosine_similarity('long', 'longer')
print(cos1)