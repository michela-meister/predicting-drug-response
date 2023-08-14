#!/bin/bash
method='transfer'
dataFn='data/yaspo_combined.csv'
writeDir='results/2023-08-13/test_pdx_expt'
foldFn='fold_info/pdx_folds.pkl'
splitSeed='1'
nSteps='5'

mkdir -p "$writeDir"

python3 code/pdx_expt.py method="$method" dataFn="$dataFn" writeDir="$writeDir" foldFn="$foldFn" splitSeed="$splitSeed" nSteps="$nSteps"