import PyPDF2
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import itertools


def get_data_pdf_files(folder_path):
        all_text = []
        file_count=len(listdir(folder_path))
        assert file_count>0
        for file in tqdm(listdir(folder_path),total=file_count):
            try:
                file_path=join(folder_path,file)
                if isfile(file_path):
                    file_obj = open(file_path, 'rb')
                    pdf_reader = PyPDF2.PdfFileReader(file_obj)
                    page_count = pdf_reader.numPages
                    for x in range(page_count):
                        pageObj = pdf_reader.getPage(x)
                        page_data=pageObj.extractText()
                        page_data_split=page_data.split("\n")
                        all_text_per_page = []
                        for line in page_data_split:
                            if len(line)>1:
                                all_text_per_page.append(line.strip().lower())
                        all_text_per_page_str=" ".join(all_text_per_page)
                        all_text.append(all_text_per_page_str)
            except Exception:
                print(Exception)
                continue
        assert len(all_text) > 0
        return all_text

def get_data(filename):
    return open(filename,"r",newline='\n')






