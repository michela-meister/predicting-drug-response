# model params; change job time if applicable!
n_mcmc=5
n_warmup=5
n_iter=3 # TODO: remove!
thinning=5
k=3
seed=$1

bd=/work/tansey/meisterm
day_dir=$bd/results/"$(date +"%Y-%m-%d")"
base_dir=$bd/results/2023-07-05/clean_data
save_dir=$day_dir/model_splits

mkdir -p $day_dir
mkdir -p $save_dir
mkdir -p $base_dir/$seed/split

python3 $bd/code/test_evaluation.py data_fn=$base_dir n_total_obs=-1 n_mcmc=$n_mcmc n_warmup=$n_warmup n_iter=$n_iter thinning=$thinning directory=$save_dir/$s \
use_real_data=0 k=$k seed=$seed
