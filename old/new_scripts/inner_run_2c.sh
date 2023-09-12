#!/bin/bash
mm="/work/tansey/meisterm"
baseDir="$mm/results/2023-08-11/inner_run_2c"
scriptDir="$mm/new_scripts"
codeDir="$mm/code"
splitType="random_split"
dataFn="$mm/data/rep-gdsc-ctd2-mean-log.csv"
foldFn=""
nSteps="1000"

# Datasets
datasetList=("REP" "GDSC" "CTD2")
datasetPairs=("REP GDSC" "REP CTD2" "GDSC CTD2" "GDSC REP" "CTD2 GDSC" "CTD2 REP")

percentList=("20" "40" "60" "80")
kList=("5" "10" "15" "20" "25" "30" "35" "40" "45" "50")

# Run target_only on individual datasets
method="target_only"
sourceData=""
for target in "${datasetList[@]}"
do
	for percent in "${percentList[@]}"
	do
		holdoutFrac=".$percent"
		for splitSeed in $(eval echo "{0..9}")
		do
			for innerSeed in $(eval echo "{0..4}")
			do
				for k in "${kList[@]}"
				do
					for modelSeed in $(eval echo "{0..4}")
					do
						writeDir="$baseDir/$target/$percent/$splitSeed/$innerSeed/$k/$modelSeed"
						mkdir -p "$writeDir"
						"$scriptDir/"inner_run_model.sh "$codeDir" "$method" "$sourceData" "$target" "$splitType" "$holdoutFrac" "$dataFn" "$writeDir" \
						"$foldFn" "$splitSeed" "$innerSeed" "$modelSeed" "$k" "$nSteps"
					done
				done
			done
		done
	done
done

# Run transfer on pairs of datasets
method="target_only"
for pair in "${datasetPairs[@]}"
do
	set -- $pair
	sourceData="$1"
	target="$2"
	echo "$sourceData" and "$target"
	for percent in "${percentList[@]}"
	do
		holdoutFrac=".$percent"
		for splitSeed in $(eval echo "{0..9}")
		do
			for innerSeed in $(eval echo "{0..4}")
			do
				for k in "${kList[@]}"
				do
					for modelSeed in $(eval echo "{0..4}")
					do
						writeDir="$baseDir/$target/$percent/$splitSeed/$innerSeed/$k/$modelSeed"
						mkdir -p "$writeDir"
						"$scriptDir/"inner_run_model.sh "$codeDir" "$method" "$sourceData" "$target" "$splitType" "$holdoutFrac" "$dataFn" "$writeDir" \
						"$foldFn" "$splitSeed" "$innerSeed" "$modelSeed" "$k" "$nSteps"
					done
				done
			done
		done
	done
done



