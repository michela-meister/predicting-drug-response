#!/bin/bash
baseDir="results/2023-08-07/opt_k_target_only/"

experiments=("REP" "GDSC" "CTD2")

for expt in "${experiments[@]}"
do
	exptDir=$baseDir"/log_"$expt
	mkdir -p "$exptDir"
	rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/results/2023-08-07/opt_k_target_only/log_"$expt"/best_k.pkl "$exptDir" 
	rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/results/2023-08-07/opt_k_target_only/log_"$expt"/diagonal.png "$exptDir"
	rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/results/2023-08-07/opt_k_target_only/log_"$expt"/test_avg.pkl "$exptDir"
	rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/results/2023-08-07/opt_k_target_only/log_"$expt"/train_avg.pkl "$exptDir"
done