#!/bin/bash
mm="/work/tansey/meisterm"
baseDir="$mm/results/2023-08-09/tt_run_2d"
scriptDir="$mm/new_scripts"
codeDir="$mm/code"
splitType="random_split"
dataFile="data/rep-gdsc-ctd2-mean-log.csv"
foldFile=""
nSteps="5"

mkdir -p $baseDir

# Run on raw, transfer
datasetPairs=("REP GDSC" "REP CTD2" "GDSC CTD2" "GDSC REP" "CTD2 GDSC" "CTD2 REP")
methodList=("raw" "transfer")
percentList=("10" "20" "30" "40" "50" "60" "70" "80" "90")

for pair in "${datasetPairs[@]}"
do
	set -- $pair
	source="$1"
	target="$2"
	echo "$source" and "$target"
	for method in "${methodList[@]}"
	do
		for percent in "${percentList[@]}"
		do
			for splitSeed in $(eval echo "{0..19}")
			do
				writeDir="$baseDir""/""log""_""$source""_""$target""/""$percent""/""$method/$splitSeed"
				holdoutFrac=".$percent"
				mkdir -p "$writeDir"
				bsub -n 1 -W 12:00 -R 'span[hosts=1] rusage[mem=32]' -e "$writeDir/err.err" -o "$writeDir/out.out" \
				"$scriptDir/"expt.sh "$codeDir" "$method" "$source" "$target" "$splitType" "$holdoutFrac" "$dataFile" "$writeDir" "$foldFile" \
				"$nSteps" "$splitSeed"
			done
		done
	done
done

# Run on target_only
method="target_only"
datasets=("REP" "GDSC" "CTD2")

for target in "${datasets[@]}"
do
	for percent in "${percentList[@]}"
	do
		for splitSeed in $(eval echo "{0..19}")
		do
			writeDir="$baseDir""/""log""_""$target""/""$percent""/""$method/$splitSeed"
			holdoutFrac=".$percent"
			mkdir -p "$writeDir"
			"$scriptDir/"expt.sh "$codeDir" "$method" "$source" "$target" "$splitType" "$holdoutFrac" "$dataFile" "$writeDir" "$foldFile" \
			"$nSteps" "$splitSeed"
		done
	done
done
