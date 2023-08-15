#!/bin/bash
codeDir="$1"
method="$2"
source=$3
target="$4" 
splitType="$5" 
holdoutFrac="$6" 
dataFile="$7"
writeDir="$8"
foldFile="$9"
nSteps="${10}"
splitSeed="${11}"

python3 "$codeDir"/raw_expt.py method="$method" source="$source" target="$target" splitType="$splitType" holdoutFrac="$holdoutFrac" dataFile="$dataFile" \
writeDir="$writeDir" foldFile="$foldFile" nSteps="$nSteps" splitSeed="$splitSeed"