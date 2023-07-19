#!/bin/bash
dataset1=REP
dataset2=GDSC
suffix='_published_auc_mean'
day_dir=results/"$(date +"%Y-%m-%d")"
save_dir=$day_dir/transfer_multi_eval/$dataset1'_'$dataset2

mkdir -p $save_dir

for r in {1..2}
do
	for k in {1..2}
	do
		mkdir -p $save_dir/$r/$k
		for s in {1..2}
		do
			python3 code/transfer_multi_eval.py seed=$s k=$k r=$r obs_name1=$dataset1$suffix obs_name2=$dataset2$suffix save_dir=$save_dir/$r/$k nsteps=5
		done
	done
done



# python3 code/digest_transfer_multi_eval.py