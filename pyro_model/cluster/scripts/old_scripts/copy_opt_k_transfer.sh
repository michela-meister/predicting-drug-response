#!/bin/bash
baseDir="results/2023-08-07/opt_k_transfer/"

datasetPairs=("REP_GDSC" "REP_CTD2" "GDSC_CTD2" "GDSC_REP" "CTD2_GDSC" "CTD2_REP")

for pair in "${datasetPairs[@]}"
do
	exptDir=$baseDir"/log_"$pair
	mkdir -p "$exptDir"
	rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/results/2023-08-07/opt_k_transfer/log_"$pair"/best_k.pkl "$exptDir" 
	rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/results/2023-08-07/opt_k_transfer/log_"$pair"/diagonal.png "$exptDir"
	rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/results/2023-08-07/opt_k_transfer/log_"$pair"/train_avg.pkl "$exptDir"
	rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/results/2023-08-07/opt_k_transfer/log_"$pair"/test_avg.pkl "$exptDir"
done
