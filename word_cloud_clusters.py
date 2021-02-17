from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np
# Start with loading all necessary libraries
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

dict_concepts={}
dict_concepts["rice "]=1
dict_concepts["rice hullers"]=1
dict_concepts["rice productivity"]=100
dict_concepts["more rice"]=1

oval_mask = np.array(Image.open("img/oval.png"))
# Create and generate a word cloud image:
wordcloud = WordCloud(prefer_horizontal=1,width=400,height=400,
                      background_color='white',
                        #max_font_size=500,
                      #min_font_size=4
                      ).generate_from_frequencies(dict_concepts)

# Display the generated image:
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()