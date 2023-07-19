#!/bin/bash
dataset1=REP
dataset2=GDSC
suffix='_published_auc_mean'
day_dir=results/"$(date +"%Y-%m-%d")"
save_dir=$day_dir/transfer_multi_eval/$dataset1'_'$dataset2

mkdir -p $save_dir


for k in {1..1}
do
	for s in {1..1}
	do
		python3 code/transfer_multi_eval.py seed=$s k=$k r=3 obs_name1=$dataset1$suffix obs_name2=$dataset2$suffix save_dir=$save_dir nsteps=5
	done
done

# python3 code/digest_transfer_multi_eval.py