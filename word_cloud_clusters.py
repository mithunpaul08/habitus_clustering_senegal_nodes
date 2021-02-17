from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np
# Start with loading all necessary libraries
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import sys
from numpy import random
from glove_read_get_embed import read_file_python_way


dataset_cluster_namees=pd.read_csv('./outputs_from_clara/cluster_names.csv')
cluster_ids = dataset_cluster_namees.iloc[:, [0]].values
cluster_names = dataset_cluster_namees.iloc[:, [1]].values
assert len(cluster_ids)== len(cluster_names)
dict_clusterid_names={}
for i,n in zip(cluster_ids,cluster_names):
    dict_clusterid_names[n[0]]=i[0]


list_dict_concepts=[]
cluster_members=read_file_python_way('./outputs_from_clara/cluster_members.csv')
for index,line in enumerate(cluster_members):
    dict_concepts = {}
    if not index ==0:
        line = line.strip()
        split_line=line.split("\t")
        id=split_line[0]
        members=split_line[1:]
        if(len(members)>3):
            for each_member in members:
                if not each_member in ["","\n"]:
                    if each_member in dict_clusterid_names:
                        dict_concepts[each_member]=1000
                    else:
                        dict_concepts[each_member] = 1
            list_dict_concepts.append(dict_concepts)



def display_wordcloud(list_dict_concepts, title, n_components):
    plt.figure(figsize=(40, 10), facecolor='white')
    j=n_components
    #j = np.ceil(n_components/4)
    for t in range(n_components):
        #i=t+1
        i = n_components
        index = random.randint(1, n_components*n_components)
        plt.subplot(j, i, index)
        plt.plot()
        oval_mask = np.array(Image.open("img/phploeBuh.png"))

        wordcloud = WordCloud(prefer_horizontal=1, width=444, height=444,
                              mask=oval_mask,
                              background_color='white',
                              contour_width=1, contour_color='white',
                              max_font_size=500,
                              # min_font_size=4
                              ).generate_from_frequencies(list_dict_concepts[t])


        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
    plt.show()




display_wordcloud(list_dict_concepts,"name of plot", len(list_dict_concepts))
sys.exit()

oval_mask = np.array(Image.open("img/phploeBuh.png"))


# Create and generate a word cloud image:
wordcloud = WordCloud(prefer_horizontal=1,width=1025,height=1000,
                      mask=oval_mask,
                      background_color='white',
contour_width=5, contour_color='black',
                        #max_font_size=500,
                      #min_font_size=4
                      ).generate_from_frequencies(dict_concepts)

# Display the generated image:
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()