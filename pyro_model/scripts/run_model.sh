#!/bin/bash
write_dir=results/"$(date +"%Y-%m-%d")"/run_model/

mkdir -p $write_dir

python3 code/fit_model.py read_fn='data/split/train.pkl' write_dir=$write_dir