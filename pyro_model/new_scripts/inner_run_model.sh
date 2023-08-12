#!/bin/bash
codeDir="$1"
method="$2"
sourceData="$3"
target="$4"
splitType="$5"
holdoutFrac="$6"
dataFn="$7"
writeDir="$8"
foldFn="$9"
splitSeed="${10}"
innerSeed="${11}"
modelSeed="${12}"
k="${13}"
nSteps="${14}"

python3 "$codeDir/"inner_run_model.py method="$method" source="$sourceData" target="$target" split_type="$splitType" holdout_frac="$holdoutFrac" \
data_fn="$dataFn" write_dir="$writeDir" fold_fn="$foldFn" split_seed="$splitSeed" inner_seed="$innerSeed" model_seed="$modelSeed" k="$k" n_steps="$nSteps"