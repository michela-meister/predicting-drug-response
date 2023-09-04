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

import expt
import cross_val
import helpers
import model_helpers as modeling

# Returns column names in dataset for given method
def get_column_names(method, source_name, target_name):
    suffix = '_auc_overlap_mean'
    if method == 'raw':
        # use published mean auc as raw baseline
        prefix = ''
    elif method == 'transfer' or method == 'target_only':
        # use log(published mean auc) for ML models
        prefix = 'log_'
    source_col = prefix + source_name + suffix
    target_col = prefix + target_name + suffix
    return source_col, target_col

def main():
    method, source_name, target_name, split_type, holdout_frac, data_fn, write_dir, fold_fn, split_seed, n_steps = expt.get_raw_args(sys.argv, 10)
    source_col, target_col = get_column_names(method, source_name, target_name)
    # Split target dataset into train and test, get number of samples and drugs in dataset
    target_train_df, target_test_df, n_samp, n_drug = helpers.get_target(data_fn, fold_fn, target_col, split_seed, holdout_frac, split_type)
    # ================================
    # MAKE PREDICTIONS BY METHOD
    assert method == 'raw'
    if method == 'raw':
        target_train_sd = helpers.get_sample_drug_ids(target_train_df)
        target_test_sd = helpers.get_sample_drug_ids(target_test_df)
        source_df = helpers.get_source(data_fn, source_col)
        train_predictions, test_predictions = expt.predict_raw(source_df, source_col, target_train_sd, target_test_sd)
        train_corr = expt.evaluate_correlation(train_predictions, target_train_df, target_col)
        test_corr = expt.evaluate_correlation(test_predictions, target_test_df, target_col)
    # ================================
    # EVALUATE PREDICTIONS AND SAVE
    # save to file
    helpers.write_pickle(train_corr, write_dir + '/train.pkl')
    helpers.write_pickle(test_corr, write_dir + '/test.pkl')
    #expt.save_predictions(write_dir + '/train_predictions.pkl', train_predictions, target_train_df)
    #expt.save_predictions(write_dir + '/test_predictions.pkl', test_predictions, target_test_df)
    # TODO: save predictions!!

    print('train_corr: ' + str(train_corr))
    print('test_corr: ' + str(test_corr))

if __name__ == "__main__":
    main()