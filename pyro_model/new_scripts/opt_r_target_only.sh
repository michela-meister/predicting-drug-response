#!/bin/bash
baseDir="results/$(date +%F)/opt_r_target_only"
scriptDir="new_scripts"
codeDir="code"
method="target_only"
source=""
# target defined below
holdoutFrac=".1"
dataFile="data/rep-gdsc-ctd2-mean-log.csv"
foldFile=""
hypFile=""
# splitSeed is defined below, based on iterating seeds
# modelSeed is defined below, based on iteration
# k defined below
r="-1"
nSteps="1000"


datasetList=("REP 37" "GDSC 48" "CTD2 32")
numSeeds=20
lastSeed="$(($numSeeds-1))"
numModels=10
lastModel="$(($numModels-1))"

for pair in "${datasetList[@]}"
do
	set -- $pair
	target="$1"
	k="$2"
	datasetDir="$baseDir/log""_""$target"
	mkdir -p "$datasetDir"
	for r in $(eval echo "{1..$k}")
	do
		for splitSeed in $(eval echo "{0..$lastSeed}")
		do 
			for modelSeed in $(eval echo "{0..$lastModel}")
			do
				writeDir="$datasetDir/$k/$r/$splitSeed/$modelSeed"
				mkdir -p "$writeDir"
				"$scriptDir/"raw.sh "$codeDir" "$method" "$source" "$target" "$holdoutFrac" "$dataFile" "$writeDir" "$foldFile" "$hypFile" \
				"$splitSeed" "$modelSeed" "$k" "$r" "$nSteps"
			done
		done
	done
done