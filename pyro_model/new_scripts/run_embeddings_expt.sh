#!/bin/bash
mm="/work/tansey/meisterm"
baseDir="$mm/results/2023-08-12/embeddings_expt"
scriptDir="$mm/new_scripts"
codeDir="$mm/code"
dataFile="data/rep-gdsc-ctd2-mean-log.csv"
nSteps="1000"

mkdir -p $baseDir

datasetPairs=("REP GDSC" "REP CTD2" "GDSC CTD2" "GDSC REP" "CTD2 GDSC" "CTD2 REP")

for pair in "${datasetPairs[@]}"
do
	set -- $pair
	sourceData="$1"
	target="$2"
	echo "$sourceData" and "$target"
	writeDir="$baseDir""/""log""_""$sourceData""_""$target"
	mkdir -p "$writeDir"
	bsub -n 1 -W 4:00 -R 'span[hosts=1] rusage[mem=32]' -e "$writeDir/err.err" -o "$writeDir/out.out" \
	"$scriptDir/"embeddings_expt.sh "$codeDir" "$sourceData" "$target" "$dataFn" "$writeDir" "$nSteps"
done  