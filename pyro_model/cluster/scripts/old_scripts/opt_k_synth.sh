#!/bin/bash
baseDir="results/$(date +%F)/opt_k_synth"
scriptDir="new_scripts"
codeDir="code"
method="target_only"
source=""
# target defined below
holdoutFrac=".1"
dataFile="data/synth.csv"
foldFile=""
hypFile=""
# splitSeed is defined below, based on iterating seeds
# modelSeed is defined below, based on iteration
# k defined below
r="-1"
nSteps="5"


datasetList=("synth")
lastK="80"
numSeeds=20
lastSeed="$(($numSeeds-1))"
numModels=10
lastModel="$(($numModels-1))"

for target in "${datasetList[@]}"
do
	datasetDir="$baseDir/log""_""$target"
	mkdir -p "$datasetDir"
	for k in $(eval echo "{1..$lastK}")
	do
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