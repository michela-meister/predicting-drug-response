#!/bin/bash
mm="/work/tansey/meisterm"
codeDir="$mm/code"
scriptDir="$mm/new_scripts"
method="transfer"
sourceData="REP"
target="GDSC"
splitType="random_split"
holdoutFrac=".1"
dataFile="$mm/data/rep-gdsc-ctd2-mean-log.csv"
writeDir="$mm/results/2023-08-10/test_transfer_8"
foldFile="$mm/fold_info/fold_list_10.pkl"
nSteps="1000"
splitSeed="0"

mkdir -p "$writeDir"

bsub -n 1 -W 8:00 -R 'span[hosts=1] rusage[mem=32]' -e "$writeDir/err.err" -o "$writeDir/out.out" \
"$scriptDir/"expt.sh "$codeDir" "$method" "$sourceData" "$target" "$splitType" "$holdoutFrac" "$dataFile" "$writeDir" "$foldFile" \
"$nSteps" "$splitSeed"

