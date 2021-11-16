import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram


DISTANCE_THRESHOLD_CLUSTERING=0.9
NO_OF_CLUSTERS=2

beliefs=[
    "work.Like working.",
    "So wried wwent xalisae xtremecaffeine",
    " yildirim z/28 zipout zulchzulu"
# "loans are useful.",
#     "Trump",
#     "Paul makes good beer",
#     "Dogfish Head makes better beer",
#     "Paul makes the best beer",
#     "there will be more virulent mutations",
#     "premises of Social Darwinism",
# "Troeg’s might know a thing or two about beer",
# "Rudy Giuliani broke the law",
# "anyone would hate The Princess Bride"
]

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

embeddings_dict = load_glove_model("glove_bottom10.txt")
print(embeddings_dict.keys())

words_indices={}
words_embeddings=[]
for belief in beliefs:
    belif_split=belief.split(" ")
    for index,each_word in enumerate(belif_split):
        if each_word in embeddings_dict:
            words_indices[index]=each_word
            words_embeddings.append(embeddings_dict[each_word])

print(words_indices)


#the engine part which does habitus and plotting. will need cosine similarities of each concept as input
model=AgglomerativeClustering(n_clusters=NO_OF_CLUSTERS, linkage='average', affinity='cosine',compute_distances=True)
model =model.fit(words_embeddings)
plot_dendrogram(model, truncate_mode="level", p=NO_OF_CLUSTERS)

# labels=model.labels_
# cluster_count=clustering.n_clusters_

# for index,clusterid in enumerate(labels.items):
#    plt.scatter(words_embeddings[labels==index, 0], words_embeddings[labels==index, 1], s=50)

######plot dendrograms
# plt.figure(figsize=(15, 12))
# dendo=sch.dendrogram(sch.linkage(words_embeddings,method='average'))
plt.show()



