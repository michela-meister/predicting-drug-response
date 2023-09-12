#!/bin/bash
codeDir="code"
readFn="data/rep-gdsc-ctd2.csv"
writeDir="data"

python3 "$codeDir"/clean_cell_line_data.py readFn="$readFn" writeDir="$writeDir"
