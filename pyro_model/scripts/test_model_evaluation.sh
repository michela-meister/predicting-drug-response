#!/bin/bash
day_dir=results/"$(date +"%Y-%m-%d")"
base_dir=$day_dir/clean_and_split_data/split

mkdir -p $day_dir
mkdir -p $base_dir

python3 code/test_model_evaluation.py train_fn=$base_dir/train.pkl test_fn=$base_dir/test.pkl sample_fn=$base_dir/sample_dict.pkl \
drug_fn=$base_dir/drug_dict.pkl