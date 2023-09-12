#!/bin/bash
day_dir=results/"$(date +"%Y-%m-%d")"
base_dir=results/2023-06-13/clean_and_split_data
save_dir=$day_dir/test_evaluation

mkdir -p $save_dir
mkdir -p $base_dir/split

for s in {1..5}
do
	mkdir -p $save_dir/$s
	python3 code/test_evaluation.py data_fn=$base_dir n_total_obs=-1 n_mcmc=5 n_warmup=5 n_iter=3 thinning=5 directory=$save_dir/$s use_real_data=0 k=3 seed=$s
done