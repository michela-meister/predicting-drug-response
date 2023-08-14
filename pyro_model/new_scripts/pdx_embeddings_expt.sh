#!/bin/bash
codeDir="code"
dataFn="data/yaspo_combined.csv"
writeDir="results/2023-08-13/test_pdx_embed"
nSteps="5"

mkdir -p "$writeDir"

python3 "$codeDir/"pdx_embeddings_expt.py data_fn="$dataFn" write_dir="$writeDir" n_steps="$nSteps"
