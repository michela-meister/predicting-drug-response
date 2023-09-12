#!/bin/bash
baseDir="results/$(date +%F)/raw_2c"
scriptDir="new_scripts"
codeDir="code"
method="raw"
source="REP"
target="GDSC"
holdoutFrac="-1"
dataFile="data/rep-gdsc-ctd2-mean-log.csv"
foldFile="fold_info/fold_list.pkl"
hypFile=""
# splitSeed is defined below, based on the fold
modelSeed="-1"
k="-1"
r="-1"
nSteps=5

datasetPairs=("REP GDSC" "REP CTD2" "GDSC CTD2" "GDSC REP" "CTD2 GDSC" "CTD2 REP")
numFolds=20
lastFold="$(($numFolds-1))"

for pair in "${datasetPairs[@]}"
do
	set -- $pair
	source="$1"
	target="$2"
	echo "$source" and "$target"
	for fold in $(eval echo "{0..$lastFold}")
	do
		echo "$fold"
		splitSeed="$fold"
		writeDir="$baseDir""/""log""_""$source""_""$target""/""$fold"
		mkdir -p "$writeDir"
		"$scriptDir/"raw.sh "$codeDir" "$method" "$source" "$target" "$holdoutFrac" "$dataFile" "$writeDir" "$foldFile" "$hypFile" \
		"$splitSeed" "$modelSeed" "$k" "$r" "$nSteps"
	done
done
