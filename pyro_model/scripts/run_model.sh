#!/bin/bash
run_dir=results/"$(date +"%Y-%m-%d")"/run_model
eval_dir=results/"$(date +"%Y-%m-%d")"/eval_model

mkdir -p $run_dir
mkdir -p $eval_dir

python3 code/fit_model.py read_fn='results/2023-05-31/clean_and_split_data/split/train.pkl' write_dir=$run_dir
python3 code/eval_ppd.py samples_fn='results/2023-06-01/run_model/mcmc_samples.pkl' test_fn='results/2023-05-31/clean_and_split_data/split/test.pkl' write_dir=$eval_dir