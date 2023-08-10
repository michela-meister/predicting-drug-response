#!/bin/bash
date
codeDir="code"
method="raw"
source="REP"
target="GDSC"
splitType="random_split"
holdoutFrac=".8"
dataFile="data/rep-gdsc-ctd2-mean-log.csv"
writeDir="results/$(date +%F)/test_transfer"
foldFile="fold_info/fold_list.pkl"
nSteps="1000"
splitSeed="0"

mkdir -p "$writeDir"

python3 "$codeDir"/expt.py method="$method" source="$source" target="$target" splitType="$splitType" holdoutFrac="$holdoutFrac" dataFile="$dataFile" \
writeDir="$writeDir" foldFile="$foldFile" nSteps="$nSteps" splitSeed="$splitSeed"

date
