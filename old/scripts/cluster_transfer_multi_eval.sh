#!/bin/bash
dataset1=REP
dataset2=GDSC
suffix='_published_auc_mean'
n_steps=5

bd=/work/tansey/meisterm
day_dir=$bd/results/"$(date +"%Y-%m-%d")"
save_dir=$day_dir/transfer_multi_eval/$dataset1'_'$dataset2

mkdir -p $save_dir

for r in {1..2}
do
	for k in {1..2}
	do
		for s in {1..2}
		do
			mkdir -p $save_dir/$r/$k/$s
			for m in {1..2}
			do
				bsub -n 1 -W 4:00 -R 'span[hosts=1] rusage[mem=32]' -e $save_dir/$r/$k/$s/model_split.err -o $save_dir/$r/$k/$s/model_split.out \
				$bd/scripts/transfer_multi_eval_inputs.sh s=$s m=$m k=$k r=$r obs_name1=$dataset1$suffix obs_name2=$dataset2$suffix save_dir=$save_dir/$r/$k/$s nsteps=$n_steps
			done
		done
	done
done

#python3 code/digest_transfer_multi_eval.py data_dir=$save_dir r_max=$r_max k_max=$k_max s_max=$s_max m_max=$m_max