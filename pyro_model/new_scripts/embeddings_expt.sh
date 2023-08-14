#!/bin/bash
codeDir=$1
sourceData=$2
target=$3
dataFn=$4
writeDir=$5
nSteps=$6

python3 "$codeDir/"embeddings_expt.py source="$sourceData" target="$target" data_fn="$dataFn" write_dir="$writeDir" n_steps="$nSteps"
