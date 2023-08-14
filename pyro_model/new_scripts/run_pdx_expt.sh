#!/bin/bash
mm="/work/tansey/meisterm"
baseDir="$mm/results/2023-08-13/pdx_expt"
scriptDir="$mm/new_scripts"
codeDir="$mm/code"
dataFn="$mm/data/yaspo_combined.csv"
foldFn="$mm/fold_info/pdx_folds.pkl"
nSteps="1000"

mkdir -p "$baseDir"

for splitSeed in {0..11}
do
	# Run script on raw
	method="raw"
	writeDir="$baseDir/$method/$splitSeed"
	mkdir -p "$writeDir"
	bsub -n 1 -W 0:30 -R 'span[hosts=1] rusage[mem=32]' -e "$writeDir/err.err" -o "$writeDir/out.out" \
	"$scriptDir/"pdx_expt.sh "$codeDir" "$method" "$dataFn" "$writeDir" "$foldFn" "$splitSeed" "$nSteps"

	# Run script on transfer
	method="transfer"
	writeDir="$baseDir/$method/$splitSeed"
	mkdir -p "$writeDir"
	bsub -n 1 -W 4:00 -R 'span[hosts=1] rusage[mem=32]' -e "$writeDir/err.err" -o "$writeDir/out.out" \
	"$scriptDir/"pdx_expt.sh "$codeDir" "$method" "$dataFn" "$writeDir" "$foldFn" "$splitSeed" "$nSteps"
done
