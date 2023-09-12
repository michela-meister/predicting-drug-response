#!/bin/bash
cluster_base_dir=/work/tansey/meisterm/results/2023-07-31/heatmap
local_base_dir=results/2023-07-31/heatmap

sh ./scripts/data_from_cluster.sh 'REP' 'GDSC' $cluster_base_dir $local_base_dir
sh ./scripts/data_from_cluster.sh 'REP' 'CTD2' $cluster_base_dir $local_base_dir
sh ./scripts/data_from_cluster.sh 'GDSC' 'REP' $cluster_base_dir $local_base_dir
sh ./scripts/data_from_cluster.sh 'GDSC' 'CTD2' $cluster_base_dir $local_base_dir
sh ./scripts/data_from_cluster.sh 'CTD2' 'REP' $cluster_base_dir $local_base_dir
sh ./scripts/data_from_cluster.sh 'CTD2' 'GDSC' $cluster_base_dir $local_base_dir



