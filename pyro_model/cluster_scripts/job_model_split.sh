conda activate
screen -S splits

bd=/work/tansey/meisterm
day_dir=$bd/results/"$(date +"%Y-%m-%d")"
base_dir=$bd/results/2023-07-05/clean_data
save_dir=$day_dir/model_splits

for s in {1..20}
do
	mkdir -p $save_dir/$s
	bsub -n 1 -W 0:30 -R 'span[hosts=1] rusage[mem=8]' -e $save_dir/$s/model_split.err -o $save_dir/$s/model_split.out /work/tansey/meisterm/cluster_scripts/model_split.sh $s
done