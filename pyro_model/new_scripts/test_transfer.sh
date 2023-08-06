#!/bin/bash
codeDir="code"
method="transfer"
source="REP"
target="GDSC"
holdoutFrac=".2"
dataFile="data/rep-gdsc-ctd2-mean-log.csv"
writeDir="results/$(date +%F)/test_transfer"
foldFile=""
hypFile=""
splitSeed="0"
modelSeed="1"
k="10"
r="10"
nSteps="1000"

mkdir -p "$writeDir"

python3 "$codeDir"/raw.py method="$method" source="$source" target="$target" holdoutFrac="$holdoutFrac" dataFile="$dataFile" writeDir="$writeDir" \
foldFile="$foldFile" hypFile="$hypFile" splitSeed="$splitSeed" modelSeed="$modelSeed" k="$k" r="$r" nSteps="$nSteps"