#!/bin/bash
day_dir=results/"$(date +"%Y-%m-%d")"
base_dir=results/2023-06-13/clean_and_split_data
save_dir=$day_dir/pipeline

mkdir -p $save_dir
mkdir -p $base_dir/split

python3 code/pipeline.py data_fn=$base_dir n_total_obs=-1 n_mcmc=200 n_warmup=500 n_iter=3 thinning=100 directory=$save_dir use_real_data=1 k=2