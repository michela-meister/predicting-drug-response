#!/bin/bash
baseDir="results/$(date +%F)/opt_k_transfer"
scriptDir="new_scripts"
codeDir="code"
method="transfer"
# source defined below
# target defined below
holdoutFrac=".1"
dataFile="data/rep-gdsc-ctd2-mean-log.csv"
foldFile=""
hypFile=""
# splitSeed is defined below, based on iterating seeds
# modelSeed is defined below, based on iteration
# k defined below
# r defined below, equal to k
nSteps="5"


datasetPairs=("REP GDSC" "REP CTD2" "GDSC CTD2" "GDSC REP" "CTD2 GDSC" "CTD2 REP")
numFolds=20
lastFold="$(($numFolds-1))"
lastK="80"
numSeeds=20
lastSeed="$(($numSeeds-1))"
numModels=10
lastModel="$(($numModels-1))"

for pair in "${datasetPairs[@]}"
do
	set -- $pair
	source="$1"
	target="$2"
	echo "$source" and "$target"
	datasetDir="$baseDir/log""_""$source""_""$target"
	mkdir -p "$datasetDir"
	for k in $(eval echo "{1..$lastK}")
	do
		r="$k"
		for splitSeed in $(eval echo "{0..$lastSeed}")
		do 
			for modelSeed in $(eval echo "{0..$lastModel}")
			do
				writeDir="$datasetDir/$k/$splitSeed/$modelSeed"
				mkdir -p "$writeDir"
				"$scriptDir/"raw.sh "$codeDir" "$method" "$source" "$target" "$holdoutFrac" "$dataFile" "$writeDir" "$foldFile" "$hypFile" \
				"$splitSeed" "$modelSeed" "$k" "$r" "$nSteps"
			done
		done
	done
done