from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt,mpld3
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
import mpld3
from mpld3 import plugins
GRID_EDGES_WIDTH=10
GRID_EDGES_HEIGHT=8


# Define some CSS to control our custom labels
css = """
table
{
  border-collapse: collapse;
}
th
{
  color: #ffffff;
  background-color: #000000;
}
td
{
  background-color: #cccccc;
}
table, th, td
{
  font-family:Arial, Helvetica, sans-serif;
  border: 1px solid black;
  text-align: right;
}
"""


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
        if(len(members)>1):
            for each_member in members:
                if not each_member in ["","\n"]:
                    if each_member in dict_clusterid_names:
                        dict_concepts[each_member]=100000
                    else:
                        dict_concepts[each_member] = 1
            list_dict_concepts.append(dict_concepts)
#senegal_flag.png


def display_wordcloud(list_dict_concepts, title, n_components):
    fig=plt.figure(figsize=(40, 10), facecolor='white')
    #fig, ax = plt.subplots()
    #j=i=GRID_EDGES
    labels=[]

    for t in range(n_components):
        labels.append(t)
        index = random.randint(1, GRID_EDGES_WIDTH*GRID_EDGES_HEIGHT)
        plt.subplot(GRID_EDGES_WIDTH, index, subplot_kw=dict(projection='polar'))
        plt.plot()

        oval_mask = np.array(Image.open("img/phploeBuh.png"))
        no_of_cluster_members=len(list_dict_concepts[t])
        wordcloud = WordCloud(prefer_horizontal=1, width=444, height=444,
                              mask=oval_mask,
                              background_color='white',
                              contour_width=5, contour_color='red',
                              max_font_size=no_of_cluster_members*500,
                              min_font_size=4,
                              ).generate_from_frequencies(list_dict_concepts[t])


        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
    plt.show()
    # points = ax.plot(100,100, 'o', color='b',
    #                  mec='k', ms=15, mew=1, alpha=.6)
    # tooltip = plugins.PointHTMLTooltip(points[0], labels,
    #                                    voffset=10, hoffset=10, css=css)
    # plugins.connect(fig, tooltip)
    #
    # mpld3.show()



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