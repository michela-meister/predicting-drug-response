#!/bin/bash
day_dir=results/"$(date +"%Y-%m-%d")"
base_dir=results/2023-06-13/clean_and_split_data/split 
save_dir=$day_dir/test_evaluation

mkdir -p $save_dir

python3 code/test_evaluation.py train_fn=$base_dir/train.pkl test_fn=$base_dir/test.pkl sample_fn=$base_dir/sample_dict.pkl drug_fn=$base_dir/drug_dict.pkl \
n_total_obs=-1 n_mcmc=10 n_warmup=10 n_iter=5 directory=$save_dir