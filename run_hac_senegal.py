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
import os
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


DISTANCE_THRESHOLD_CLUSTERING=0.3
SIMILARITY_THRESHOLD=0.7

#list of queries that were taken from tomek's model, and were in turn used to run queries on google to download the pdf files from which CONCEPTS were extracted using odin
QUERIES_AKA_VARIABLES =[
    "dummy value village information",
    "Food Insecurity" ,
   "Agricultural yield" ,
    "Fertilizers" ,
    "Fertilizers" ,
    "Pesticides" ,
    "Flood timing" ,
    "Credit worthiness time" ,
    "Loan interest rate" ,
    "Planting time" ,
    "Personal Capital" ,
    "Field Size" ,
    "Workforce quality" ,
    "Weather" ,
    "Economic features" ,
    "Weather" ,
    "Food Insecurity" ,
    "Economic features" ,
    "Fertilizers" ,
    "Fertilizers" ,
    "Pesticides" ,
    "Workforce age" ,
    "Flood timing" ,
    "Credit worthiness time" ,
    "Loan interest rate" ,
    "Planting time" ,
    "Loan availability" ,
    "Loan availability",
     "Agricultural profit",
]

random.seed(3)

#read causes and effects
dataset=pd.read_csv('./senegal_africa.csv')
causes = dataset.iloc[:, [2]].values

#usee this if you want effeects  column
effects = dataset.iloc[:, [5]].values

combined_causes_effects=[]


for cause in causes:
    combined_causes_effects.append(cause[0])
for effect in effects:
    combined_causes_effects.append(effect[0])

#get only unique values out
combined_causes_effects=set(combined_causes_effects)
combined_causes_effects=list(combined_causes_effects)


def write_to_csv(data,filename):
    folder_file=os.path.join("outputs",filename)
    with open(folder_file,'w',newline='') as myfile:
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

def normalize(vector):
    norms=np.apply_along_axis(np.linalg.norm,0,vector)
    return vector/norms

def split_concept_get_average_embedding(concept_name):
    emb_total=np.full([1,300],0.00000001)
    cause_split_tokens = concept_name.split()
    total_no_of_tokens=len(cause_split_tokens)

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
    emb_total=emb_total/total_no_of_tokens #divide by number of tokens to get average embedding for this concept
    return emb_total



emb1=None
emb2=None
for index,(concepts) in enumerate(combined_causes_effects):
    concept_name=concepts
    all_embeddings=split_concept_get_average_embedding(concept_name)
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
model=AgglomerativeClustering(n_clusters=None, distance_threshold=DISTANCE_THRESHOLD_CLUSTERING, linkage='average', compute_full_tree=True, affinity='cosine')
clustering =model.fit(all_data)
labels=model.labels_
cluster_count=clustering.n_clusters_

print(f"total number of concepts is {index}")
print(f"total number of clusters is {cluster_count}")
#print(f"final labels are  {labels}")
#print(f"distances are{clustering.distances_}")

# plot the dendrogram before clustering process






#to map which cluster did finally each concept end up
concept_text_cluster_id={}
clusterid_to_concept_text={}



for index,label in enumerate(labels):
    concept_name=map_id_to_concept_name.get(index,None)
    assert concept_name is not None
    concept_text_cluster_id[concept_name]=label

    #to get tfhe list of concepts clusterd under same cluster id
    list_concepts_under_this_id=clusterid_to_concept_text.get(label,None)
    if list_concepts_under_this_id is None:
        new_list=[concept_name]
        clusterid_to_concept_text[label]=new_list
    else:
        assert type(list_concepts_under_this_id) is list
        list_concepts_under_this_id.append(concept_name)
        clusterid_to_concept_text[label] = list_concepts_under_this_id


filename='concept_clusterid_threshold' + str(DISTANCE_THRESHOLD_CLUSTERING) + ".csv"
assert len(concept_text_cluster_id.keys()) > 0
write_to_csv(concept_text_cluster_id,filename)

#to find the namee of the ]cluster

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

'''
Qn) how to name the new clusters you are creating?

Ans:/pick the concept that’s closest to the avg embedding of the cluster: 
For example: (women, female, wives ) PROMOTE (production, rice growing)
emb(women) closest to avg(emb(women), emb(female), emb(wives)) => “women” becomes the name of the cluster
women 17'''

def given_multi_token_concept_get_average_embedding(cluster_of_concepts):
    emb_all_concepts_for_this_clusterid = []
    for each_concept in cluster_of_concepts:
        emb_of_each_concept=split_concept_get_average_embedding(each_concept)
        emb_all_concepts_for_this_clusterid.append(emb_of_each_concept)
    return emb_all_concepts_for_this_clusterid

def get_average_emb_of_a_cluster(cluster_of_concepts):
    average_emb_all_concepts_for_this_clusterid = given_multi_token_concept_get_average_embedding(cluster_of_concepts)
    avg_emb_of_this_cluster_of_concepts = sum(average_emb_all_concepts_for_this_clusterid) / len(cluster_of_concepts)
    return avg_emb_of_this_cluster_of_concepts

cluster_id_cluster_name={}
#go through the dict clusterid_to_concept_text. for each key, get all the list of names. for each name, calculate its embeddings, pick the embedding which is in the middle and call it the name of the cluster
for cluster_id, cluster_of_concepts in clusterid_to_concept_text.items():
    avg_emb_of_this_cluster_of_concepts=get_average_emb_of_a_cluster(cluster_of_concepts)




    #all clusters will be temporarily given the name of the first guy in th cluster. this is to test how good teh clusters are before we go itno naming them
    # todo, fix+uncomment thesee
    # index_of_element_closest_to_average=find_nearest(average_emb_all_concepts_for_this_clusterid,avg_of_concept_names)
    #cluster_id_cluster_name[cluster_id] = list_concepts[index_of_element_closest_to_average]
    cluster_id_cluster_name[cluster_id]=cluster_of_concepts[0]


assert len(clusterid_to_concept_text.keys()) > 0
filename='clusterid_to_all_sub_concepts_threshold' + str(DISTANCE_THRESHOLD_CLUSTERING) + ".csv"
write_to_csv(clusterid_to_concept_text,filename)



filename='cluster_id_cluster_name_threshold' + str(DISTANCE_THRESHOLD_CLUSTERING) + ".csv"
assert len(cluster_id_cluster_name.keys()) > 0
write_to_csv(cluster_id_cluster_name,filename)

#all plotting related stuff
##### plot clusters
# for k,v in clusterid_to_concept_text.items():
#    plt.scatter(all_data[labels==k, 0], all_data[labels==k, 1], s=50)
######plot dendrograms
#plt.figure(figsize=(15, 12))
#dendo=sch.dendrogram(sch.linkage(all_data,method='average'))
#plt.show()


'''check if all the query variables exist in concepts, or atleast are close in embedding space
# This is to check the efficacy of our clustering algo. i.e of the starting queries, how many were we able to retrieve back, in full or atleast close to it in embedding space.
steps:
-.  
take each of the query we used initially in google search,..
- find its embedding.
- if the phrase exists as is in a cluster, we are done..

else:
- now go through each of these clusters
- go through each of the sub concepts in each cluster
- get average embedding for each sub concept
- get average embedding for a given cluster

- get a cosine similarity betweeen average embedding of a given cluster with teh average embedding of each query. 
- If they have a cosine similarity of greater than similarity threshold, then add them to a list...
- then find the cluster with the highest similarity score to the given query.
'''

def find_best_matching_cluster_for_a_given_query(clusterid_to_concept_text, query_variable):
    emb_query_variable = split_concept_get_average_embedding(query_variable)
    clusterid_to_cosine_sim_value_with_query={}

    best_cosine_sim_value_below_similarity_threshold=0
    best_cluster_id_below_similarity_threshold = 0

    for cluster_id,cluster in  clusterid_to_concept_text.items():
        total_no_of_concepts_in_this_cluster = len(cluster)
        all_emb_of_concepts_in_a_cluster = []
        for each_sub_concept in cluster:
            if each_sub_concept.lower()==query_variable.lower(): #check if there is an exact string match.
                #print(f"found exact match between a query variable and a concept == {query_variable}")
                return cluster_id , clusterid_to_concept_text[cluster_id],0, True

        #else:sum of all embeddings of all concepts in a cluster, divided by the number of concepts in a cluster. note that this is not a scalar value but instead an embedding itself
        average_embedding_of_a_cluster=get_average_emb_of_a_cluster(cluster)

        #- now get a cosine similarity betweeen average embedding of a given cluster with teh average embedding of each query.
        cos=cosine_similarity(average_embedding_of_a_cluster, emb_query_variable)

        #- If they have a cosine similarity of greater than similarity threshold, then add thAT cosine sim value  to a dict{cluster_id,cosine similarity value }...
        if cos > SIMILARITY_THRESHOLD:
            clusterid_to_cosine_sim_value_with_query[cluster_id]=cos
        if cos > best_cosine_sim_value_below_similarity_threshold:
            best_cosine_sim_value_below_similarity_threshold=cos
            best_cluster_id_below_similarity_threshold=cluster_id
    best_cluster_below_similarity_threshold=clusterid_to_concept_text[best_cluster_id_below_similarity_threshold]


    best_cosine_sim_value = 0
    best_cluster_cluster_id = 0

    #- then find the cluster with the highest similarity score to the given query.
    if len(clusterid_to_cosine_sim_value_with_query.keys()) > 0 :
        for k,v in clusterid_to_cosine_sim_value_with_query.items():
            if v > best_cosine_sim_value:
                best_cosine_sim_value = v
                best_cluster_cluster_id=k
    else:
        print(f"There was no match with any cluster for this query {query_variable}, which was more than a similarity threshold of 0.7. However the "
              f"beest match was with "
              f"the cluster with clusterid={best_cluster_id_below_similarity_threshold} , the highest value of cosine sim was {cos}and members:{best_cluster_below_similarity_threshold}")

    return best_cluster_cluster_id, clusterid_to_concept_text[best_cluster_cluster_id],best_cosine_sim_value, False





for query_variable in QUERIES_AKA_VARIABLES:
    cluster_id_of_best_match_cluster, best_match_cluster,best_cosine_sim_value,found_string_match=find_best_matching_cluster_for_a_given_query(clusterid_to_concept_text, query_variable)

    if found_string_match:
        print(
            f"Found a direct string match. Closest cluster for the given query variable: {query_variable} : is cluster id:{cluster_id_of_best_match_cluster} . Also it"
            f" had an exact string match with a concept")

    else:
            if type(best_cosine_sim_value) == np.ndarray:
                best_cosine_sim_value=best_cosine_sim_value[0][0]
            print(
                f"Found a matching cluster to the query with similarity threshold > 0.7. query is: {query_variable} ..and cluster id "
                f"is:{cluster_id_of_best_match_cluster} with a cosine sim value of {best_cosine_sim_value}. "
                f" The concepts in that cluster are:"
                f"{best_match_cluster}")




