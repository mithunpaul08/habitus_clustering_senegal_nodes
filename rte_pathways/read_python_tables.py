from os import listdir
from os.path import join,isfile
import camelot
from tqdm import tqdm
from utils import get_no_of_pages_in_a_pdf_file

def read_all_pdf_files_in_folder():
    folder_path = "data/pdffiles/"
    file_count = len(listdir(folder_path))
    assert file_count > 0
    for file in tqdm(listdir(folder_path), total=file_count, desc="reading google crawled files:"):
        if ".pdf" in file:
            file_path = join(folder_path, file)
            read_tables(file_path)

def read_tables(file_path):
                if isfile(file_path):
                    no_of_pages=get_no_of_pages_in_a_pdf_file(file_path)
                    print(f"{file_path}")
                    for page in range(no_of_pages):
                        list_pages = []
                        list_pages.append(str(page))
                        str_list_pages=",".join(list_pages)
                        tables = camelot.read_pdf(file_path,pages=str_list_pages)
                        if len(tables) > 0:
                            for each_table in tables:
                                print(f"table in page number {page}")
                                df=each_table.df
                                print(df)


#read_all_pdf_files_and_their_tables()

read_tables('data/pdffiles/05.pdf')
tables=camelot.read_pdf('data/pdffiles/05.pdf',flavor='stream')
assert len(tables) > 0
df = tables[0].df
print(df)
print(tables[0].parsing_report)

