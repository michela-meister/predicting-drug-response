#!/bin/bash
day_dir=results/"$(date +"%Y-%m-%d")"
base_dir=results/2023-06-13/clean_and_split_data

mkdir -p $day_dir/split

python3 code/split_train_test.py in_path=$base_dir/welm_pdx_clean_mid_volume.csv write_dir=$day_dir/split 
