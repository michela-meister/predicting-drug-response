#!/bin/bash
dataset1=$1
dataset2=$2
prefix=$3
suffix=$4
n_steps=$5
k_max=$6
s_max=$7
m_max=$8
data_dir=$9
save_dir=$10

mkdir -p $save_dir

for k in $(eval echo "{1..$k_max}")
do
    for r in $(eval echo "{1..$k}")
    do
		for s in $(eval echo "{1..$s_max}")
		do
			mkdir -p $save_dir/$r/$k/$s
			for m in $(eval echo "{1..$m_max}")
			do
				python3 code/transfer_multi_eval.py s=$s m=$m k=$k r=$r obs_name1=$prefix$dataset1$suffix obs_name2=$prefix$dataset2$suffix save_dir=$save_dir/$r/$k/$s \
				nsteps=$n_steps data_dir=$data_dir
			done
		done
	done
done