#!/bin/bash
codeDir="/work/tansey/meisterm/code"
sMax="20"
mMax="10"

baseDir="/work/tansey/meisterm/results/2023-08-07/opt_r_target_only"

datasetList=("REP 37" "GDSC 48" "CTD2 32")
numSeeds=20
lastSeed="$(($numSeeds-1))"
numModels=10
lastModel="$(($numModels-1))"

for pair in "${datasetList[@]}"
do
	set -- $pair
	target="$1"
	kMax="$2"
	resultsDir="$baseDir/log""_""$target"
	python3 "$codeDir/"digest_opt_r.py resultsDir="$resultsDir" kMax="$kMax" sMax="$sMax" mMax="$mMax"
done


