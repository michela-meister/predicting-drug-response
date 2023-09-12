#!/bin/bash
mm="/work/tansey/meisterm"
baseDir="$mm/results/2023-08-14/pdx_embeddings_expt"
scriptDir="$mm/new_scripts"
codeDir="$mm/code"
dataFile="$mm/data/yaspo_combined_intersection.csv"
nSteps="1000"

mkdir -p $baseDir

writeDir="$baseDir"
bsub -n 1 -W 4:00 -R 'span[hosts=1] rusage[mem=32]' -e "$writeDir/err.err" -o "$writeDir/out.out" \
"$scriptDir/"pdx_embeddings_expt.sh "$codeDir" "$dataFile" "$writeDir" "$nSteps"
