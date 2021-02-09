import sklearn
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import sys
from glove_read_get_embed import get_embedding_given_token

#read causes and effects
dataset=pd.read_csv('./senegal_africa.csv')
causes = dataset.iloc[:, [2]].values

# effects = dataset.iloc[:, [5]].values
# assert causes.shape == effects.shape

#for each token in cause and effect, get their glove embedding
#Average embeddings for multi-word concepts


total_oov_words=0
total_tokens=0
concept_emb={}
X=[]
for cause in causes:
    concept_name=cause[0]
    cause_split_tokens=concept_name.split()
    total=[]
    z = np.zeros(300)
    emb_total= z.reshape(1,-1)

    #for each sub token, get embedding of them, and get the average of all n token embeddings as the concepts overall embedding value
    for each_token in cause_split_tokens:
        total_tokens+=1
        emb,flag_oov=get_embedding_given_token(each_token)
        emb_total+=emb
        if (flag_oov==True):
            total_oov_words+=1
    avg_emb=np.average(emb_total)
    concept_emb[concept_name]=avg_emb
    X.append(avg_emb)



print(f"total number of oov tokens/total tokens={total_oov_words/total_tokens}")

X=np.asarray(X).reshape(1,-1)
#the engine part which does clustering and plotting. will need cosine similarities of each concept as input
model=AgglomerativeClustering(n_clusters=None, distance_threshold=0.3, linkage='average',compute_full_tree=True)
model.fit(X)
labels=model.labels_

plt.scatter(X[labels==0, 0], X[labels==0, 1], s=50, marker='o', color='red')
plt.scatter(X[labels==1, 0], X[labels==1, 1], s=50, marker='o', color='blue')
plt.scatter(X[labels==2, 0], X[labels==2, 1], s=50, marker='o', color='green')
plt.scatter(X[labels==3, 0], X[labels==3, 1], s=50, marker='o', color='purple')
plt.scatter(X[labels==4, 0], X[labels==4, 1], s=50, marker='o', color='orange')
plt.show()


# plt.figure(figsize=(15, 12))
# dendo=sch.dendrogram(sch.linkage(X,method='average'))
# plt.show()
