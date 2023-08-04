#!/bin/bash
codeDir="code"
dataFile="data/rep-gdsc-ctd2-mean-log.csv"
foldFile=""
writeDir="results/$(date +%F)/test_raw"
source="REP"
target="GDSC"
splitSeed=0
holdoutFrac=.5

mkdir -p "$writeDir"

python3 "$codeDir"/raw.py dataFile="$dataFile" foldFile="$foldFile" writeDir="$writeDir" source="$source" target="$target" splitSeed="$splitSeed" holdoutFrac="$holdoutFrac"