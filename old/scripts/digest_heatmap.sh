#!/bin/bash
dataset1=$1
dataset2=$2
results_dir=$3
k_max=$4
s_max=$5
m_max=$6

python3 code/digest_heatmap.py results_dir=$results_dir k_max=$k_max s_max=$s_max m_max=$m_max