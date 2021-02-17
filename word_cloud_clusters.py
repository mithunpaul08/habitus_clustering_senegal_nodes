from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np
# Start with loading all necessary libraries
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

def transform_format(val):
    if val == 0:
        return 255
    else:
        return val

dict_concepts={}
dict_concepts["rice "]=1
dict_concepts["rice hullers"]=1
dict_concepts["rice productivity"]=100
dict_concepts["more rice"]=1

oval_mask = np.array(Image.open("img/phploeBuh.png"))
# transformed_wine_mask = np.ndarray((oval_mask.shape[0],oval_mask.shape[1]), np.int32)
# for i in range(len(oval_mask)):
#     transformed_wine_mask[i] = list(map(transform_format, oval_mask[i]))


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