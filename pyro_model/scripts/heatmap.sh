#!/bin/bash
dataset1=REP
dataset2=GDSC
suffix='_published_auc_mean'
n_steps=5

day_dir=results/"$(date +"%Y-%m-%d")"
save_dir=$day_dir/heatmap/$dataset1'_'$dataset2

mkdir -p $save_dir

k_max=3
s_max=1
m_max=1

for k in $(eval echo "{1..$k_max}")
do
    for r in $(eval echo "{1..$k}")
    do
		for s in $(eval echo "{1..$s_max}")
		do
			mkdir -p $save_dir/$r/$k/$s
			for m in $(eval echo "{1..$m_max}")
			do
				python3 code/transfer_multi_eval.py s=$s m=$m k=$k r=$r obs_name1=$dataset1$suffix obs_name2=$dataset2$suffix save_dir=$save_dir/$r/$k/$s nsteps=$n_steps
			done
		done
	done
done

#python3 code/digest_transfer_multi_eval.py data_dir=$save_dir r_max=$r_max k_max=$k_max s_max=$s_max m_max=$m_max