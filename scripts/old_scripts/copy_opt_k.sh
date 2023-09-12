#!/bin/bash
# COPY TARGET_ONLY RESULTS
baseDir="results/2023-08-07/opt_k_target_only"

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

# COPY TRANSFER RESULTS
baseDir="results/2023-08-07/opt_k_transfer"

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

# COPY SYNTH RESULTS
baseDir="results/2023-08-07/opt_k_synth"

datasetPairs=("synth")

for pair in "${datasetPairs[@]}"
do
	exptDir=$baseDir"/log_"$pair
	mkdir -p "$exptDir"
	rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/results/2023-08-07/opt_k_synth/log_"$pair"/best_k.pkl "$exptDir" 
	rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/results/2023-08-07/opt_k_synth/log_"$pair"/diagonal.png "$exptDir"
	rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/results/2023-08-07/opt_k_synth/log_"$pair"/train_avg.pkl "$exptDir"
	rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/results/2023-08-07/opt_k_synth/log_"$pair"/test_avg.pkl "$exptDir"
done