#!/bin/bash
dataset1=$1
dataset2=$2
cluster_dir=$3/log_$dataset1'_'$dataset2
local_dir=$4/log_$dataset1'_'$dataset2

mkdir -p $local_dir

declare -a files=("/train_avg.pkl" "/test_avg.pkl" "/heatmap_train.png" "/heatmap_test.png" "/heatmap_all.pdf" "/fixed_r.pdf" "/fixed_k.pdf")

for f in "${files[@]}"
do
	echo $f
	rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:$cluster_dir$f $local_dir
done