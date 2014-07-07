#!/usr/bin/env sh
# This script create the file_list to give to the prototxt

mkdir temp

cd pettorine
find -type f -name '*' | sed -n '19601,28000p' | cut -c18-21 > ../temp/labels_true.txt
find `pwd` -type f -name '*' | sed -n '19601,28000p' > ../temp/filenames_true.txt
paste -d ' ' ../temp/filenames_true.txt ../temp/labels_true.txt > ../temp/file_listTEST.txt 

cd ../no_pettorine
find `pwd` -type f -name '*' | sed -n '21,40p' > ../temp/filenames_false.txt
awk '{print $0 " 10000"}' ../temp/filenames_false.txt  >> ../temp/file_listTEST.txt

echo "Done."
