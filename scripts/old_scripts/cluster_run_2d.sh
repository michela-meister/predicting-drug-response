#!/bin/bash
baseDir="experiment2"
scriptDir="scripts"
codeDir="code"
splitType="random_split"
dataFile="data/rep-gdsc-ctd2-mean-log.csv"
foldFile=""
nSteps="1000"

mkdir -p $baseDir

datasetPairs=("REP GDSC" "REP CTD2" "GDSC CTD2" "GDSC REP" "CTD2 GDSC" "CTD2 REP")
percentList=("10" "20" "30" "40" "50" "60" "70" "80" "85" "90" "95")

method="raw"
for pair in "${datasetPairs[@]}"
do
        set -- $pair
        sourceData="$1"
        target="$2"
        echo "$sourceData" and "$target"
        for percent in "${percentList[@]}"
        do
                for splitSeed in $(eval echo "{0..9}")
                do
                        writeDir="$baseDir""/""log""_""$sourceData""_""$target""/""$percent""/""$method/$splitSeed"
                        holdoutFrac=".$percent"
                        mkdir -p "$writeDir"
                        bsub -n 1 -W 0:30 -R 'span[hosts=1] rusage[mem=32]' -e "$writeDir/err.err" -o "$writeDir/out.out" \
                        "$scriptDir/"expt.sh "$codeDir" "$method" "$sourceData" "$target" "$splitType" "$holdoutFrac" "$dataFile" "$writeDir" "$foldFile" \
                        "$nSteps" "$splitSeed"
                done
        done
done

method="transfer"
for pair in "${datasetPairs[@]}"
do
        set -- $pair
        sourceData="$1"
        target="$2"
        echo "$sourceData" and "$target"
        for percent in "${percentList[@]}"
        do
                for splitSeed in $(eval echo "{0..9}")
                do
                        writeDir="$baseDir""/""log""_""$sourceData""_""$target""/""$percent""/""$method/$splitSeed"
                        holdoutFrac=".$percent"
                        mkdir -p "$writeDir"
                        bsub -n 1 -W 24:00 -R 'span[hosts=1] rusage[mem=32]' -e "$writeDir/err.err" -o "$writeDir/out.out" \
                        "$scriptDir/"expt.sh "$codeDir" "$method" "$sourceData" "$target" "$splitType" "$holdoutFrac" "$dataFile" "$writeDir" "$foldFile" \
                        "$nSteps" "$splitSeed"
                done
        done
done

# Run on target_only
method="target_only"
datasets=("REP" "GDSC" "CTD2")
sourceData=""

for target in "${datasets[@]}"
do
        for percent in "${percentList[@]}"
        do
                for splitSeed in $(eval echo "{0..9}")
                do
                        writeDir="$baseDir""/""log""_""$target""/""$percent""/""$method/$splitSeed"
                        holdoutFrac=".$percent"
                        mkdir -p "$writeDir"
                        bsub -n 1 -W 24:00 -R 'span[hosts=1] rusage[mem=32]' -e "$writeDir/err.err" -o "$writeDir/out.out" \
                        "$scriptDir/"expt.sh "$codeDir" "$method" "$sourceData" "$target" "$splitType" "$holdoutFrac" "$dataFile" "$writeDir" "$foldFile" \
                        "$nSteps" "$splitSeed"
                done
        done
done
