#!/bin/bash
codeDir="code"
method="target_only"
sourceData="REP"
target="GDSC"
splitType="random_split"
holdoutFrac=".1"
dataFn="data/rep-gdsc-ctd2-mean-log.csv"
writeDir="results/2023-08-11/inner_transfer_test"
foldFn=""
splitSeed="0"
innerSeed="2"
modelSeed="3"
k="10"
nSteps="1000"

mkdir -p "$writeDir"

new_scripts/inner_run_model.sh "$codeDir" "$method" "$sourceData" "$target" "$splitType" "$holdoutFrac" "$dataFn" "$writeDir" \
"$foldFn" "$splitSeed" "$innerSeed" "$modelSeed" "$k" "$nSteps"