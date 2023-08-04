#!/bin/bash
scriptDir="new_scripts"
codeDir="code"
dataFile="data/rep-gdsc-ctd2-mean-log.csv"
foldFile="fold_info/fold_list.pkl"
baseDir="results/$(date +%F)/raw_2c"
holdoutFrac=-1 # because doing k-fold

datasetPairs=("REP GDSC" "REP CTD2" "GDSC CTD2" "GDSC REP" "CTD2 GDSC" "CTD2 REP")
maxFold=20

for pair in "${datasetPairs[@]}"
do
	set -- $pair
	source="$1"
	target="$2"
	echo "$source" and "$target"
	for fold in $(eval echo "{1..$maxFold}")
	do
		echo "$fold"
		writeDir="$baseDir/log_$source""_""$target/$fold"
		mkdir -p "$writeDir"
		"$scriptDir/"raw.sh "$codeDir" "$dataFile" "$foldFile" "$writeDir" "$source" "$target" "$fold" "$holdoutFrac" 
	done
done
