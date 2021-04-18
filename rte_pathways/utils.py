import PyPDF2
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import pdftotext


def read_txt_data(folder_path):
    all_text = []
    file_count = len(listdir(folder_path))
    assert file_count > 0
    for file in tqdm(listdir(folder_path), total=file_count, desc="reading google crawled files:"):
        try:
            if ".txt" in file:
                file_path = join(folder_path, file)
                if isfile(file_path) :
                        try:
                            file = open(file_path)
                            google_crawled_data=file.read()
                            all_text.append(google_crawled_data)
                        except Exception:
                            print(Exception)
                            continue
        except Exception:
            print(Exception)
            continue
    assert len(all_text) > 0
    return all_text



def get_data(filename):
    return open(filename,"r",newline='\n')




# sample text for human description, pruned versions of which are found in data/human_description.txt
# The main qualitative difference between the clusters is that cluster A has one peak, cluster C has two peaks and cluster C has no peaks. This interpretation requires no knowledge, just the ability to see patterns in wiggly lines.  Going further:
#
# In pathway A, the fields lie fallow during the rainy season. The NDVI index shows no greening because little or nothing is grown.  The slight downward trend in the NDVI index might be due to preparing the land and/or flooding, both of which would produce a lower NDVI index.  Planting happens around month 6 with rapid maturation over the subsequent two months.  Grain filling happens during month 8 and harvest begins around month 9.
#
# In pathway C ,land preparation occurs in months 1-3, followed by steady growth of the crop, peaking around month 5.  At this point the crop is mature and grain filling happens during month 5, after which harvest begins.  A very quick land preparation phase begins around month 7 and a second crop is planted.  The second crop grows from month 8 to month 11.
#
# In pathway B, little or nothing is grown during the rainy season. The small increase in greening during month 8 might be due to dry-season market garden cropping.
#
# << In fact, pathway B happens in the part of the valley that’s far from the river, so it’s unclear how much water they get. It’s also unclear whether any crops are grown or whether the observed increase in greening is due to indigenous plants. >>
#
# All of which tells me that the interpretation of clusters is a knowledge-intensive activity! I intentionally did not elide concepts like “the rainy season” and “lie fallow” and “grain filling”.







