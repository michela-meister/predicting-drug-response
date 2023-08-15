#!/bin/bash
codeDir=$1
method=$2
dataFn=$3
writeDir=$4
foldFn=$5
splitSeed=$6
nSteps=$7

python3 "$codeDir/"pdx_expt.py method="$method" dataFn="$dataFn" writeDir="$writeDir" foldFn="$foldFn" splitSeed="$splitSeed" nSteps="$nSteps"