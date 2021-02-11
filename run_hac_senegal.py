import sklearn
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import sys
from glove_read_get_embed import get_embedding_given_token
import csv
import cfg
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


THRESHOLD=0.6

random.seed(3)

#read causes and effects
dataset=pd.read_csv('./senegal_africa.csv')
causes = dataset.iloc[:, [2]].values

#usee this if you want effeects instead of causes column
effects = dataset.iloc[:, [5]].values

combined_causes_effects=[]


for cause in causes:
    combined_causes_effects.append(cause[0])
for effect in effects:
    combined_causes_effects.append(effect[0])


# effects = dataset.iloc[:, [5]].values
# assert causes.shape == effects.shape


def write_to_csv(data,filename):
    with open(filename,'w',newline='') as myfile:
        for k,v in  data.items():
            row=([k,v])
            mywriter=csv.writer(myfile,delimiter='\t')
            mywriter.writerow(row)



# for each token in cause and effect, get their glove embedding
# Average embeddings for multi-word concepts

concept_emb={}
all_data=np.zeros([1,300])
map_concept_name_to_id={}
map_id_to_concept_name={}


def split_concept_get_combined_embedding(concept_name):
    #z = np.zeros(300)
    #emb_total = z.reshape(1,-1)

    emb_total=np.full([1,300],0.00000001)
    cause_split_tokens = concept_name.split()
    #for each sub token, get embedding of them, and get the average of all n token embeddings as the concepts overall embedding value
    for each_token in cause_split_tokens:
        if each_token in stopwords.words('english'):
            continue
        cfg.total_tokens+=1
        emb_raw=get_embedding_given_token(each_token)
        if emb_raw is None:
            cfg.total_oov_words += 1
            continue
        else:
            emb_total+=emb_raw

#todo: do normalization by embedding by dividing with total magnitude of the vector
    if np.count_nonzero(emb_total[0])==0:
        return None
    else:
        return emb_total[0]



emb1=None
emb2=None
for index,(concepts) in enumerate(combined_causes_effects):
    concept_name=concepts
    all_embeddings=split_concept_get_combined_embedding(concept_name)
    if all_embeddings is not None:
        concept_emb[concept_name]=all_embeddings
        map_concept_name_to_id[concept_name]=index
        map_id_to_concept_name[index]=concept_name
        all_embeddings=all_embeddings.reshape(1,-1)
        all_data=np.append(all_data,all_embeddings,axis=0)
    else:
        print(f"found that the concept at index {index} and concept={concepts} had a zero embedding")





print(f"total number of oov tokens/total tokens={str(cfg.total_oov_words)}/{str(cfg.total_tokens)}")

#remove the first row of zeros
all_data = np.delete(all_data,0,axis=0)





#the engine part which does clustering and plotting. will need cosine similarities of each concept as input
model=AgglomerativeClustering(n_clusters=None, distance_threshold=THRESHOLD, linkage='average',compute_full_tree=True,affinity='cosine')
clustering =model.fit(all_data)
labels=model.labels_
cluster_count=clustering.n_clusters_

print(f"total number of concepts is {index}")
print(f"total number of clusters is {cluster_count}")
print(f"final labels are  {labels}")
print(f"distances are{clustering.distances_}")

# plot the dendrogram before clustering process






#to map which cluster did finally each concept end up
concept_text_cluster_id={}
clusterid_to_concept_text={}



for index,label in enumerate(labels):
    concept_name=map_id_to_concept_name.get(index,None)
    assert concept_name is not None
    concept_text_cluster_id[concept_name]=label

    #to get the list of concepts clusterd under same cluster id
    list_concepts_under_this_id=clusterid_to_concept_text.get(label,None)
    if list_concepts_under_this_id is None:
        new_list=[concept_name]
        clusterid_to_concept_text[label]=new_list
    else:
        assert type(list_concepts_under_this_id) is list
        list_concepts_under_this_id.append(concept_name)
        clusterid_to_concept_text[label] = list_concepts_under_this_id


filename='concept_clusterid_threshold'+str(THRESHOLD)+".csv"
assert len(concept_text_cluster_id.keys()) > 0
write_to_csv(concept_text_cluster_id,filename)

#to find the namee of the ]cluster

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

'''
pick the concept that’s closest to the avg embedding of the cluster: For example: (women, female, wives ) PROMOTE (production, rice growing)
emb(women) closest to avg(emb(women), emb(female), emb(wives)) => “women” becomes the name of the cluster
women 17'''
cluster_id_cluster_name={}
#go through the dict clusterid_to_concept_text. for each key, get all the list of names. for each name, calculate its embeddings, pick the embedding which is in the middle and call it the name of the cluster
for cluster_id,list_concepts in clusterid_to_concept_text.items():
    average_emb_all_concepts_for_this_clusterid=[]
    for each_concept in list_concepts:
        average_emb=split_concept_get_combined_embedding(each_concept)
        average_emb_all_concepts_for_this_clusterid.append(average_emb)
    avg_of_concept_names=sum(average_emb_all_concepts_for_this_clusterid)/len(average_emb_all_concepts_for_this_clusterid)
    index_of_element_closest_to_average=find_nearest(average_emb_all_concepts_for_this_clusterid,avg_of_concept_names)


    #all clusters will be temporarily given the name of the first guy in th cluster. this is to test how good teh clusters are before we go itno naming them
    # todo, uncomment thiscluster_id_cluster_name[cluster_id]=list_concepts[index_of_element_closest_to_average]
    cluster_id_cluster_name[cluster_id]=list_concepts[0]


assert len(cluster_id_cluster_name.keys()) > 0
filename='clusterid_to_all_sub_concepts_threshold'+str(THRESHOLD)+".csv"
write_to_csv(clusterid_to_concept_text,filename)



filename='cluster_id_cluster_name_threshold'+str(THRESHOLD)+".csv"
assert len(cluster_id_cluster_name.keys()) > 0
write_to_csv(cluster_id_cluster_name,filename)

for k,v in clusterid_to_concept_text.items():
    plt.scatter(all_data[labels==k, 0], all_data[labels==k, 1], s=50)


plt.figure(figsize=(15, 12))
dendo=sch.dendrogram(sch.linkage(all_data,method='average'))
#plt.show()