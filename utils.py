import pandas as pd
import csv


def read_file(filename):
    lines=open(filename,mode='r')
    return lines


def read_csv_pandas(input_filepath):
    with open(input_filepath,"r",encoding="utf-8-sig") as filepath:
        input_data=pd.read_csv(filepath,"\t",header=None)
        return input_data.values.tolist()

def read_csv_python(input_filepath):
    all_lines=[]
    with open(input_filepath, newline='') as filepath:
        reader = csv.reader(filepath, delimiter="|")
        for row in reader:
            all_lines.append(" ".join(row))
    return all_lines

