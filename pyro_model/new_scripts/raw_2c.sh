#!/bin/bash
codeDir="bleh"
dataFile="blah"
foldFile="blah"
baseDir="blah"
holdoutFrac=-1 # because doing k-fold

datasetPairs=("REP GDSC" "REP CTD2" "GDSC CTD2" "GDSC REP" "CTD2 GDSC" "CTD2 REP")
maxFold=3

for pair in "${datasetPairs[@]}"
do
	set -- $pair
	source="$1"
	target="$2"
	echo "$source" and "$target"
	for fold in $(eval echo "{0..$maxFold}")
	do
		echo "$fold"
		writeDir="$baseDir/log_$source""_""$target/$fold"
		raw.sh "$codeDir" "$dataFile" "$foldFile" "$writeDir" "$source" "$target" "$fold" "$holdoutFrac" 
	done
done
