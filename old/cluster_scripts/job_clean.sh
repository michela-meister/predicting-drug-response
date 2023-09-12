conda activate mmenv
screen -S clean_data

bdir=/work/tansey/meisterm/results/2023-07-05
mkdir -p bdir

bsub -n 1 -W 0:30 -R 'span[hosts=1] rusage[mem=8]' -e $bdir/clean_data.err -o $bdir/clean_data.out /work/tansey/meisterm/cluster_scripts/clean_data.sh
