#!/bin/bash
day_dir=results/"$(date +"%Y-%m-%d")"
base_dir=results/2023-06-13/clean_and_split_data
save_dir=$day_dir/test_evaluation

mkdir -p $save_dir
mkdir -p $base_dir/split

#python3 code/test_evaluation.py train_fn=$base_dir/train.pkl test_fn=$base_dir/test.pkl sample_fn=$base_dir/sample_dict.pkl drug_fn=$base_dir/drug_dict.pkl \
#n_total_obs=-1 n_mcmc=5 n_warmup=5 n_iter=3 directory=$save_dir

python3 code/test_evaluation.py data_fn=$base_dir n_total_obs=-1 n_mcmc=200 n_warmup=500 n_iter=3 thinning=100 directory=$save_dir