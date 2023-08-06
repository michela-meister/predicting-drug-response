#!/bin/bash
codeDir="code"
method="raw"
source="REP"
target="GDSC"
holdoutFrac=".5"
dataFile="data/rep-gdsc-ctd2-mean-log.csv"
writeDir="results/$(date +%F)/test_raw"
foldFile=""
hypFile=""
splitSeed="0"
modelSeed="-1"
k="-1"
r="-1"

mkdir -p "$writeDir"

python3 "$codeDir"/raw.py method="$method" source="$source" target="$target" holdoutFrac="$holdoutFrac" dataFile="$dataFile" writeDir="$writeDir" \
foldFile="$foldFile" hypFile="$hypFile" splitSeed="$splitSeed" modelSeed="$modelSeed" k="$k" r="$r"