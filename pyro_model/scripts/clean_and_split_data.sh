#!/bin/bash
base_dir=results/"$(date +"%Y-%m-%d")"/clean_and_split_data

mkdir -p $base_dir
mkdir -p $base_dir/split
mkdir -p $base_dir/normality_plots

python3 code/clean_data.py read_fn=data/welm_pdx.csv write_dir=$base_dir
python3 code/add_mids.py read_fn=$base_dir/welm_pdx_clean.csv write_dir=$base_dir
python3 code/plot_mid_data.py in_path=$base_dir/welm_pdx_clean_mid.csv write_dir=$base_dir
python3 code/transform_volume.py in_path=$base_dir/welm_pdx_clean_mid.csv write_dir=$base_dir min_duration=22
python3 code/plot_normality.py in_path=$base_dir/welm_pdx_clean_mid_volume.csv write_dir=$base_dir/normality_plots min_duration=22
python3 code/split_train_test.py in_path=$base_dir/welm_pdx_clean_mid_volume.csv write_dir=$base_dir/split 
