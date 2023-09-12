import numpy as np
import pandas as pd
import pyro
import pyro.util
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
import scipy.stats
from sklearn.model_selection import KFold
import sys
import torch
import tqdm

import expt
import cross_val
import helpers
import model_helpers as modeling

# This script applies BMT model to the PDO-PDX dataset.

# Values for parameter k in PDO-PDX setting
K_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9]
N_MODELS = 5
N_SPLITS = 5

# Save parameters to file.
def save_params(method, data_fn, write_dir, fold_fn, split_seed, n_steps):
    params = {}
    params['method'] = method
    params['data_fn'] = data_fn
    params['write_dir'] = write_dir
    params['fold_fn'] = fold_fn
    params['split_seed'] = split_seed
    params['n_steps'] = n_steps
    helpers.write_pickle(params, write_dir + '/params.pkl')

# Check validity of parameters passed in.
def check_params(method, data_fn, write_dir, fold_fn, split_seed, n_steps):
    assert method in ['raw', 'transfer']
    assert fold_fn != ""
    assert split_seed in range(0, 18)
    assert n_steps >= 5

# Read in parameters.
def get_args(args, n):
    if len(args) != n + 1:
        print('Error! Expected ' + str(n + 1) + ' arguments but got ' + str(len(args)))
        return
    method = args[1].split("=")[1]
    data_fn = args[2].split("=")[1]
    write_dir = args[3].split("=")[1]
    fold_fn = args[4].split("=")[1]
    split_seed = int(args[5].split("=")[1])
    n_steps = int(args[6].split("=")[1])
    # verify that params are valid for method given and save
    check_params(method, data_fn, write_dir, fold_fn, split_seed, n_steps)
    save_params(method, data_fn, write_dir, fold_fn, split_seed, n_steps)
    return method, data_fn, write_dir, fold_fn, split_seed, n_steps

# Compute spearman correlation between predictions and dataframe column.
def spearman_corr(predictions, df, col):
    test = df[col].to_numpy()
    assert len(predictions) == len(test)
    res = scipy.stats.spearmanr(test, predictions)
    return res.correlation

# Main function for running the PDO-PDX experiment.
def main():
    source_col = 'log10_ic50_(uM)'
    target_col = 'T_C'
    split_type = 'sample_split'
    holdout_frac = -1
    method, data_fn, write_dir, fold_fn, split_seed, n_steps = get_args(sys.argv, 6)
    # Split target dataset into train and test, get number of samples and drugs in dataset
    target_train_df, target_test_df, n_samp, n_drug = helpers.get_target(data_fn, fold_fn, target_col, split_seed, holdout_frac, split_type)
    # Make predictions by method.
    assert method in ['raw', 'transfer']
    if method == 'raw':
        target_train_sd = helpers.get_sample_drug_ids(target_train_df)
        target_test_sd = helpers.get_sample_drug_ids(target_test_df)
        source_df = helpers.get_source(data_fn, source_col)
        train_predictions, test_predictions = expt.predict_raw(source_df, source_col, target_train_sd, target_test_sd)
        train_corr = expt.evaluate_correlation(train_predictions, target_train_df, target_col)
        test_corr = expt.evaluate_correlation(test_predictions, target_test_df, target_col)
    else:
        s_idx_test, d_idx_test = helpers.get_sample_drug_indices(target_test_df)
        source_df = helpers.get_source(data_fn, source_col)
        # Choose k
        k = expt.choose_k('transfer', target_train_df, target_col, split_type, n_samp, n_drug, n_steps, source_df, source_col)
        helpers.write_pickle(k, write_dir + '/k.pkl')
        # With k fixed, fit model.
        train_predict_list, test_predict_list = expt.predict_transfer_wrapper(source_df, source_col, target_train_df, target_col, s_idx_test, d_idx_test, n_samp, 
            n_drug, n_steps, k)    
        train_corr, test_corr, train_predictions, test_predictions = expt.evaluate(train_predict_list, test_predict_list, target_train_df, target_test_df, target_col)
    # Evaluate predictions and save.
    helpers.write_pickle(train_corr, write_dir + '/train.pkl')
    helpers.write_pickle(test_corr, write_dir + '/test.pkl')
    expt.save_predictions(write_dir + '/train_predictions.pkl', train_predictions, target_train_df)
    expt.save_predictions(write_dir + '/test_predictions.pkl', test_predictions, target_test_df)
    # Save spearman correlation
    train_spear = spearman_corr(train_predictions, target_train_df, target_col)
    test_spear = spearman_corr(test_predictions, target_test_df, target_col)
    helpers.write_pickle(train_spear, write_dir + '/train_spearman.pkl')
    helpers.write_pickle(test_spear, write_dir + '/test_spearman.pkl')

    print('train_corr: ' + str(train_corr))
    print('test_corr: ' + str(test_corr))

if __name__ == "__main__":
    main()
