import PyPDF2
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import itertools


def get_data_google_crawled_files(folder_path):
        all_text = []
        file_count=len(listdir(folder_path))
        assert file_count>0
        for file in tqdm(listdir(folder_path),total=file_count,desc="reading google crawled files:"):
            try:
                file_path=join(folder_path,file)
                if isfile(file_path):
                    file_obj = open(file_path, 'rb')
                    pdf_reader = PyPDF2.PdfFileReader(file_obj)
                    page_count = pdf_reader.numPages
                    for x in range(page_count):
                            try:
                                pageObj = pdf_reader.getPage(x)
                                page_data=pageObj.extractText()
                                # page_data_split=page_data.split("\n\n")
                                # all_text_per_page = []
                                # for line in page_data_split:
                                #     if len(line)>1:
                                #         all_text_per_page.append(line.strip().lower())
                                # all_text_per_page_str=" ".join(all_text_per_page)
                                all_text.append(page_data)
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







