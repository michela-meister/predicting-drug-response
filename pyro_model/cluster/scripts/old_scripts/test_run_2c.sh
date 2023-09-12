#!/bin/bash
mm="/work/tansey/meisterm"
baseDir="$mm/results/2023-08-09/tt_run_2c"
scriptDir="$mm/new_scripts"
codeDir="$mm/code"
splitType="sample_split"
holdoutFrac="-1"
dataFile="data/rep-gdsc-ctd2-mean-log.csv"
foldFile="fold_info/fold_list.pkl"
nSteps="5"

mkdir -p $baseDir

datasetPairs=("REP GDSC" "REP CTD2" "GDSC CTD2" "GDSC REP" "CTD2 GDSC" "CTD2 REP")
methodList=("raw" "transfer")

for pair in "${datasetPairs[@]}"
do
	set -- $pair
	source="$1"
	target="$2"
	echo "$source" and "$target"
	for method in "${methodList[@]}"
	do
		for splitSeed in $(eval echo "{0..19}")
		do
			writeDir="$baseDir""/""log""_""$source""_""$target""/""$method/$splitSeed"
			mkdir -p "$writeDir"
			bsub -n 1 -W 12:00 -R 'span[hosts=1] rusage[mem=32]' -e "$writeDir/err.err" -o "$writeDir/out.out" \
			"$scriptDir/"expt.sh "$codeDir" "$method" "$source" "$target" "$splitType" "$holdoutFrac" "$dataFile" "$writeDir" "$foldFile" \
			"$nSteps" "$splitSeed"
		done
	done
done




