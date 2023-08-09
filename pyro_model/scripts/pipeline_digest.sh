#!/bin/bash
dataset1=REP
dataset2=GDSC
prefix='log_'
suffix='_published_auc_mean'
n_steps=5
k_max=3
s_max=1
m_max=1
data_dir='~/Documents/research/tansey/msk_intern/pyro_model/data'
save_dir=results/"$(date +"%Y-%m-%d")"/dummy/heatmap/$prefix$dataset1'_'$dataset2


sh ./scripts/digest_heatmap.sh $dataset1 $dataset2 $save_dir $k_max $s_max $m_max