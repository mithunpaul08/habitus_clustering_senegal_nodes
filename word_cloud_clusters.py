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

def display_wordcloud(dict_concepts, title, n_components):
    #plt.figure()
    plt.figure(figsize=(40, 10), facecolor='black')
    j = np.ceil(n_components/4)
    for t in range(n_components):
        i=t+1
        plt.subplot(j, 4, i)
            #.set_title("Topic #" + str(t))
        plt.plot()
        oval_mask = np.array(Image.open("img/phploeBuh.png"))

        wordcloud = WordCloud(prefer_horizontal=1, width=4444, height=4444,
                              mask=oval_mask,
                              background_color='white',
                              contour_width=5, contour_color='black',
                              # max_font_size=500,
                              # min_font_size=4
                              ).generate_from_frequencies(dict_concepts)

        # Display the generated image:

        plt.imshow(wordcloud, interpolation='bilinear')
        # plt.axis("off")
        # plt.show()

        #plt.imshow(WordCloud().fit_words(top_words[t]))
        plt.axis("off")
    #fig.suptitle(title)
    plt.show()


dict_concepts={}
dict_concepts["rice "]=1
dict_concepts["rice hullers"]=1
dict_concepts["rice productivity"]=1330
dict_concepts["more rice"]=1


top_words={}
top_words[0]=["rice","rice","rice","rice","rice","rice",]
top_words[1]=["corn","corn","corn","corn"]
top_words[2]=["cnn","abc","nbs","fox"]
top_words[3]=["donkey","monkey","giraffe","dog","cat","lizard",]

display_wordcloud(dict_concepts,"name of plot", 4)
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