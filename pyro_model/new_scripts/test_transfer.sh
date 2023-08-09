#!/bin/bash
date
codeDir="code"
method="transfer"
source="REP"
target="GDSC"
holdoutFrac=".8"
dataFile="data/rep-gdsc-ctd2-mean-log.csv"
writeDir="results/$(date +%F)/test_transfer"
foldFile=""
#foldFile="fold_info/fold_list.pkl"
hypFile=""
splitSeed="0"
modelSeed="1"
k="25"
r="25"
nSteps="1000"

mkdir -p "$writeDir"

python3 "$codeDir"/raw.py method="$method" source="$source" target="$target" holdoutFrac="$holdoutFrac" dataFile="$dataFile" writeDir="$writeDir" \
foldFile="$foldFile" hypFile="$hypFile" splitSeed="$splitSeed" modelSeed="$modelSeed" k="$k" r="$r" nSteps="$nSteps"
date
