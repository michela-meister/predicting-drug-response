#!/bin/bash
codeDir=$1
method=$2
source=$3
target=$4
holdoutFrac=$5
dataFile=$6
writeDir=$7
foldFile=$8
hypFile=$9
splitSeed=${10}
modelSeed=${11}
k=${12}
r=${13}
nSteps=${14}

mkdir -p "$writeDir"

python3 "$codeDir"/raw.py method="$method" source="$source" target="$target" holdoutFrac="$holdoutFrac" dataFile="$dataFile" writeDir="$writeDir" \
foldFile="$foldFile" hypFile="$hypFile" splitSeed="$splitSeed" modelSeed="$modelSeed" k="$k" r="$r" nSteps="$nSteps"