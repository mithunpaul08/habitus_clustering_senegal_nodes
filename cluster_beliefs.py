import csv
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram

'''
THe first step of the entire project is extracting beliefs from spoken text. 
for example if there is a sentence which says 'i believe Paul makes good beer' the belief is "Paul makes good beer",.
This extraction of beliefs will be done using processors.
The second part is this code, which clusters each such beliefs. In the code, we take the glove vector of each of the
word in a given belief sentence, add them all together and divide by the number of words to get an average vector that
represents each such belief sentence.
'''

#paths in laptop
# BELIEFS_FILE="/Users/mordor/research/habitus/out/mentions.tsv"
# GLOVE_FILE_NAME="glove.txt"
# DISTANCE_THRESHOLD_CLUSTERING=0.1
# NO_OF_CLUSTERS=3


#paths in server
BELIEFS_FILE="/work/mithunpaul/habitus/habitus_clulab_repo_wisconsin/out/mentions.tsv"
DISTANCE_THRESHOLD_CLUSTERING=0.2
NO_OF_CLUSTERS=30
GLOVE_FILE_NAME="/work/mithunpaul/glove/glove_lemmas.840B.300d.txt"

# beliefs=[
# "loans are useful.",
#     "Paul makes good beer",
#     "Dogfish Head makes better beer",
#     "Paul makes the best beer",
#     "there will be more virulent mutations",
#     "premises of Social Darwinism",
# "Troeg’s might know a thing or two about beer",
# "Rudy Giuliani broke the law",
# "anyone would hate The Princess Bride"
# ]

beliefs=[]
tsv_file = open(BELIEFS_FILE)
read_tsv = csv.reader(tsv_file, delimiter="\t")

for row in read_tsv:
  beliefs.append(row[1])

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def load_glove_model(glove_file):
    """
    :param glove_file: embeddings_path: path of glove file.
    :return: glove model
    """
    embeddings_dict = {}
    with open(glove_file, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

embeddings_dict = load_glove_model(GLOVE_FILE_NAME)


sent_indices={}
beliefsent_embeddings=[]
for index1, belief_sentence in enumerate(beliefs):
    sentence_split=belief_sentence.split(" ")
    combined_embedding_sent=np.zeros(300)

    for index2,each_word in enumerate(sentence_split):
        if each_word in embeddings_dict:
            combined_embedding_sent=combined_embedding_sent+embeddings_dict[each_word]
    avg_emb_per_belief_sent=combined_embedding_sent/(len(sentence_split))

    #add only if the average is not a zero vector
    if not np.sum(avg_emb_per_belief_sent)==0:
        sent_indices[index1]=belief_sentence
        beliefsent_embeddings.append(avg_emb_per_belief_sent)

f= open("beliefs_indices.txt","a")

for k,v in sent_indices.items():
    f.write(f"{k}:{v}\n")


#the engine part which does habitus and plotting. will need cosine similarities of each concept as input
model=AgglomerativeClustering(n_clusters=NO_OF_CLUSTERS, linkage='average', affinity='cosine',compute_distances=True)
model =model.fit(beliefsent_embeddings)
plot_dendrogram(model, truncate_mode="level", p=NO_OF_CLUSTERS)
plt.savefig('clusters.png')



