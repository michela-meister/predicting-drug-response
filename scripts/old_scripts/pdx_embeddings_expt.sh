#!/bin/bash
codeDir=$1
dataFn=$2
writeDir=$3
nSteps=$4

echo "$dataFn"

python3 "$codeDir/"pdx_embeddings_expt.py data_fn="$dataFn" write_dir="$writeDir" n_steps="$nSteps"
