#!/bin/bash
dataset1=REP
dataset2=GDSC
suffix='_published_auc_mean'
n_steps=5

k_max=2
s_max=2
m_max=2

bd=/work/tansey/meisterm
day_dir=$bd/results/"$(date +"%Y-%m-%d")"
save_dir=$day_dir/heatmap/$dataset1'_'$dataset2

mkdir -p $save_dir

for k in $(eval echo "{1..$k_max}")
do
    for rr in $(eval echo "{1..$k}")
    do
		for s in $(eval echo "{1..$s_max}")
		do
			mkdir -p $save_dir/$rr/$k/$s
			for m in $(eval echo "{1..$m_max}")
			do
				bsub -n 1 -W 0:30 -R 'span[hosts=1] rusage[mem=32]' -e $save_dir/$rr/$k/$s/model_split.err -o $save_dir/$rr/$k/$s/model_split.out \
                                $bd/scripts/heatmap_input.sh $s $m $k $rr $dataset1$suffix $dataset2$suffix $save_dir/$rr/$k/$s $n_steps
			done
		done
	done
done

#python3 code/digest_transfer_multi_eval.py data_dir=$save_dir r_max=$r_max k_max=$k_max s_max=$s_max m_max=$m_max