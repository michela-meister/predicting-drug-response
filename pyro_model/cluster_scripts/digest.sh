#!/bin/bash
bd=/work/tansey/meisterm
read_dir=$bd/results/2023-07-05/model_splits

python3 $bd/code/digest_vec_test_evaluation.py read_dir=$read_dir n=20