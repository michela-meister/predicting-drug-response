#!/bin/bash
cluster_dir=/work/tansey/meisterm/results/2023-07-31/heatmap/REP_GDSC
local_dir='results/2023-07-31/heatmap/REP_GDSC'

mkdir -p $local_dir

declare -a files=("/train_avg.pkl" "/test_avg.pkl" "/heatmap_train.png" "/heatmap_test.png" "/heatmap_all.pdf" "/fixed_r.pdf" "/fixed_k.pdf")

for f in "${files[@]}"
do
	echo $f
	rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:$cluster_dir$f $local_dir
done