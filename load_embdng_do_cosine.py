'''
This code was written for hoang to do somee cosine similarity test
 this is just a subseet of all functions i have. Dont usee it for clustering.
'''

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from run_hac_senegal import split_concept_get_average_embedding
import sklearn
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import sys
from glove_read_get_embed import *
import csv
import cfg
from sklearn.metrics.pairwise import cosine_similarity


GLOVE_PATH_CLARA_SERVER="/data/nlp/corpora/glove/glove.840B.300d.10f.txt"
GLOVE_PATH_LOCAL="./small_glove10f.txt"



words_embeddings={}


read_from_glove()

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




for index,(concepts) in enumerate(combined_causes_effects):
    concept_name=concepts
    all_embeddings=split_concept_get_average_embedding(concept_name)
    if all_embeddings is not None:
        pass
        #do whatever- storee the embeeddings in a dictionary, and theen use this:
        #cosine_similarity(emb_word1, emb_word2)
    else:
        print(f"found that the concept at index {index} and concept={concepts} had a zero embedding")

