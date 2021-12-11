#!/usr/bin/env bash

INPUT_DIR="./data/pdffiles"
OUTPUT_DIR="./data/txtfiles/"

for complete_path in $(find $INPUT_DIR -name '*.pdf');
do
echo "going to convert the following file "
filename= $complete_path |xargs -n 1 basename
echo "value of input file is"
echo $filename
outputfilename= "$OUTPUT_DIR$filename"
echo "value of output files is"
echo $outputfilename
pdftotext $complete_path
done
