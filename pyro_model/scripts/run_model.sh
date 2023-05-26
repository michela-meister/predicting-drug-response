#!/bin/bash
write_dir=results/"$(date +"%Y-%m-%d")"/run_model/

mkdir -p $write_dir

python3 code/fit_model.py read_fn='results/2023-05-26/clean_and_split_data/split/train.pkl' write_dir=$write_dir