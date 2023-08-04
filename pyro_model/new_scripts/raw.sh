#!/bin/bash
codeDir=$1
dataFile=$2
foldFile=$3
writeDir=$4
source=$5
target=$6
splitSeed=$7
holdoutFrac=$8

python3 "$codeDir"/raw.py dataFile="$dataFile" foldFile="$foldFile" writeDir="$writeDir" source="$source" target="$target" splitSeed="$splitSeed" holdoutFrac="$holdoutFrac"