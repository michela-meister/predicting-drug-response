import numpy as np
import pandas as pd
import pyro
import pyro.util
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
from sklearn.model_selection import KFold
import sys
import torch
import tqdm

import cross_val
import helpers
import model_helpers as modeling
import expt


#============================================================================================================
#============================================================================================================ 

def get_inner_fold(data_fn, fold_fn, target_col, split_seed, holdout_frac, split_type, inner_seed):
    # get target_train_df by splitting on split_seed, discard target_test_df
    target_train_df, _, n_samp, n_drug = helpers.get_target(data_fn, fold_fn, target_col, split_seed, holdout_frac, split_type)
    # get train_df, val_df by splitting target_train_df using kFold, indexing via innerSeed
    X = cross_val.get_items_to_split(target_train_df, split_type)
    kf = KFold(n_splits=5, random_state=707, shuffle=True)
    train_index = None
    val_index = None
    for i, (train_i, val_i) in enumerate(kf.split(X)):
        if i == inner_seed:
            train_index = train_i
            val_index = val_i
    assert train_index is not None
    assert val_index is not None
    train_df, val_df = cross_val.split_dataframe(target_train_df, 'sample_id', 'drug_id', X, split_type, train_index, val_index)
    return train_df, val_df, n_samp, n_drug

def save_params(method, source, target, split_type, holdout_frac, data_fn, write_dir, fold_fn, split_seed, inner_seed, model_seed, k, n_steps):
    params = {}
    params['method'] = method
    params['source'] = source
    params['target'] = target
    params['split_type'] = split_type
    params['holdout_frac'] = holdout_frac
    params['data_fn'] = data_fn
    params['write_dir'] = write_dir
    params['fold_fn'] = fold_fn
    params['split_seed'] = split_seed
    params['inner_seed'] = inner_seed
    params['model_seed'] = model_seed
    params['k'] = k
    params['n_steps'] = n_steps
    helpers.write_pickle(params, write_dir + '/params.pkl')

def check_params(method, source, target, split_type, holdout_frac, data_fn, write_dir, fold_fn, split_seed, inner_seed, model_seed, k, n_steps):
    assert method in ['raw', 'transfer', 'target_only']
    assert target in ['REP', 'GDSC', 'CTD2']
    assert split_type in ['random_split', 'sample_split']
    assert split_seed in list(range(10))
    if method == 'target_only':
        assert split_type == 'random_split'
    if method == 'transfer' or method == 'raw':
        assert source in ['REP', 'GDSC', 'CTD2']
    if method == 'transfer' or method == 'target_only':
        assert inner_seed in list(range(5))
        assert k in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        assert n_steps == 1000
        assert model_seed in list(range(5))
    if split_type == 'random_split':
        assert 0 <= holdout_frac and holdout_frac <= 1
    if split_type == 'sample_split':
        assert fold_fn != ""

def get_args(args, n):
    if len(args) != n + 1:
        print('Error! Expected ' + str(n + 1) + ' arguments but got ' + str(len(args)))
        return
    method = args[1].split("=")[1]
    source = args[2].split("=")[1]
    target = args[3].split("=")[1]
    split_type = args[4].split("=")[1]
    holdout_frac = float(args[5].split("=")[1])
    data_fn = args[6].split("=")[1]
    write_dir = args[7].split("=")[1]
    fold_fn = args[8].split("=")[1]
    split_seed = int(args[9].split("=")[1])
    inner_seed = int(args[10].split("=")[1])
    model_seed = int(args[11].split("=")[1])
    k = int(args[12].split("=")[1])
    n_steps = int(args[13].split("=")[1])
    # check and save params
    check_params(method, source, target, split_type, holdout_frac, data_fn, write_dir, fold_fn, split_seed, inner_seed, model_seed, k, n_steps)
    save_params(method, source, target, split_type, holdout_frac, data_fn, write_dir, fold_fn, split_seed, inner_seed, model_seed, k, n_steps)
    return method, source, target, split_type, holdout_frac, data_fn, write_dir, fold_fn, split_seed, inner_seed, model_seed, k, n_steps

# RUN MODEL FOR INNER CV LOOP
# Inputs: method, target, (source), split_type, splitSeed, innerSeed, k, modelSeed
def main():
    method, source_name, target_name, split_type, holdout_frac, data_fn, write_dir, fold_fn, split_seed, inner_seed, model_seed, k, n_steps = get_args(sys.argv, 13)
    source_col, target_col = expt.get_column_names(method, source_name, target_name)
    train_df, val_df, n_samp, n_drug = get_inner_fold(data_fn, fold_fn, target_col, split_seed, holdout_frac, split_type, inner_seed)
    # run model on train_df, val_df
    s_idx_val, d_idx_val = helpers.get_sample_drug_indices(val_df)
    if method == 'target_only':
        train_predict, val_predict = expt.run_predict_target_only(model_seed, train_df, target_col, s_idx_val, d_idx_val, n_samp, n_drug, n_steps, k)
    elif method == 'transfer':
        source_df = helpers.get_source(data_fn, source_col)
        train_predict, val_predict = expt.run_predict_transfer(model_seed, source_df, source_col, train_df, target_col, s_idx_val, d_idx_val, n_samp, 
            n_drug, n_steps, k)
    train_corr = expt.evaluate_correlation(train_predict, train_df, target_col)
    val_corr = expt.evaluate_correlation(val_predict, val_df, target_col)
    # save outputs!
    helpers.write_pickle(train_corr, write_dir + '/train.pkl')
    helpers.write_pickle(val_corr, write_dir + '/val.pkl')
    # print out
    print('train_corr: ' + str(train_corr))
    print('val_corr: ' + str(val_corr))

if __name__ == "__main__":
    main()