#!/bin/bash
FILES=../data/raman/to_convert/*

for f in $FILES
do
  echo "Processing $f file..."
  sed -i -e 's/,/./g' $f
  sed -i -e 's/;/,/g' $f
done

