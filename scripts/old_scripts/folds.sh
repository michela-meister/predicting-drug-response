#!/bin/bash
codeDir="code"
dataFile="data/rep-gdsc-ctd2-mean-log.csv"
writeFile="fold_info/fold_list_10.pkl"

python3 "$codeDir"/folds.py dataFile="$dataFile" writeFile="$writeFile"