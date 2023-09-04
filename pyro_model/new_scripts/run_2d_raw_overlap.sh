#!/bin/bash
mm="/Users/michelameister/Documents/research/drug-response-repo/drug-response/pyro_model"
baseDir="$mm/results/2023-08-31/run_2d_raw_overlap"
scriptDir="$mm/new_scripts"
codeDir="$mm/code"
splitType="random_split"
dataFile="$mm/data/rep-gdsc-ctd2-mean-log.csv"
foldFile=""
nSteps="1000"

mkdir -p $baseDir

datasetPairs=("REP GDSC" "REP CTD2" "GDSC CTD2" "GDSC REP" "CTD2 GDSC" "CTD2 REP")
percentList=("10" "20" "30" "40" "50" "60" "70" "80" "85" "90" "95")

method="raw"
for pair in "${datasetPairs[@]}"
do
        echo "$pair"
        set -- $pair
        sourceData="$1"
        target="$2"
        echo "$sourceData" and "$target"
        for percent in "${percentList[@]}"
        do
                echo "$percent"
                for splitSeed in $(eval echo "{0..9}")
                do
                        writeDir="$baseDir""/""log""_""$sourceData""_""$target""/""$percent""/""$method/$splitSeed"
                        mkdir -p "$writeDir"
                        holdoutFrac=".$percent"
                        "$scriptDir/"raw_expt.sh "$codeDir" "$method" "$sourceData" "$target" "$splitType" "$holdoutFrac" "$dataFile" "$writeDir" "$foldFile" \
                        "$nSteps" "$splitSeed"
                done
        done
done
