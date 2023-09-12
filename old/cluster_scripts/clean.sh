#!/bin/bash
bd=/work/tansey/meisterm
day_dir=$bd/results/"$(date +"%Y-%m-%d")"
base_dir=$day_dir/clean_data

mkdir -p $base_dir
mkdir -p $base_dir/normality_plots

python3 $bd/code/clean_data.py read_fn=$bd/data/welm_pdx.csv write_dir=$base_dir
python3 $bd/code/add_mids.py read_fn=$base_dir/welm_pdx_clean.csv write_dir=$base_dir
python3 $bd/code/create_paper_plots.py read_fn=$base_dir/welm_pdx_clean_mid_w_excel_sheet.csv write_dir=$base_dir
python3 $bd/code/plot_relative_mid_data.py in_path=$base_dir/welm_pdx_clean_mid.csv write_dir=$base_dir
python3 $bd/code/plot_short_mid_data.py in_path=$base_dir/welm_pdx_clean_mid.csv write_dir=$base_dir
python3 $bd/code/transform_volume.py in_path=$base_dir/welm_pdx_clean_mid.csv write_dir=$base_dir min_duration=22
python3 $bd/code/plot_normality.py in_path=$base_dir/welm_pdx_clean_mid_volume.csv write_dir=$base_dir/normality_plots min_duration=22
