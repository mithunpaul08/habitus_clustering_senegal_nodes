from utils import *

if __name__=="__main__":
    # data=get_data("data/habitus_rice_growing_senegal/")
    data_human_desc = get_data("data/human_description.txt")
    data_pdfs=get_data_pdf_files("data/temp/")


    for premise in data_human_desc:
        for hyp in data_pdfs:
            print(premise)
            print(hyp)