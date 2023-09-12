#!/bin/bash
bd=/work/tansey/meisterm
read_dir=$bd/results/2023-07-05/model_splits

bsub -n 1 -W 0:30 -R 'span[hosts=1] rusage[mem=8]' -e $read_dir/digest.err -o $read_dir/digest.out /work/tansey/meisterm/cluster_scripts/digest.sh

