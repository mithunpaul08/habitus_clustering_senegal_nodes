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
                    for page in range(no_of_pages):
                        tables = camelot.read_pdf(file_path)
                        if len(tables) > 0:
                                print(f"{file_path}")
                                df=tables[0].df
                                print(df)
                                exit
                        else:
                            continue

#read_all_pdf_files_and_their_tables()

read_tables('data/pdffiles/05.pdf')
tables=camelot.read_pdf('data/pdffiles/05.pdf',flavor='stream')
assert len(tables) > 0
df = tables[0].df
print(df)
print(tables[0].parsing_report)

