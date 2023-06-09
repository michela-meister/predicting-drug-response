#!/bin/bash
day_dir=results/"$(date +"%Y-%m-%d")"
run_dir=$day_dir/run_model
eval_dir=$day_dir/eval_model
split_dir=$day_dir'/clean_and_split_data/split'

mkdir -p $day_dir
mkdir -p $run_dir
mkdir -p $eval_dir
mkdir -p $split_dir

python3 code/fit_model.py train_fn=$split_dir'/train.pkl' sample_fn=$split_dir'/sample_dict.pkl' drug_fn=$split_dir'/drug_dict.pkl' write_dir=$run_dir
#python3 code/evaluate_model.py mcmc_samples_fn=$run_dir'/mcmc_samples.pkl' test_fn=$split_dir'/test.pkl' write_dir=$eval_dir