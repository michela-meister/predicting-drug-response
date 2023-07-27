#!/bin/bash
dataset1=REP
dataset2=GDSC
results_dir=results/2023-07-27/heatmap/$dataset1'_'$dataset2
k_max=2
s_max=2
m_max=2

python3 code/digest_heatmap.py results_dir=$results_dir k_max=$k_max s_max=$s_max m_max=$m_max