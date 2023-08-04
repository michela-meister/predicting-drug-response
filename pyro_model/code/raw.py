import numpy as np
import pandas as pd
import pyro.util
import sys

import helpers

def get_raw_args(args, n):
    if len(args) != n + 1:
        print('Error! Expected ' + str(n + 1) + ' arguments but got ' + str(len(args)))
        return
    data_fn = args[1].split('=')[1]
    fold_fn = args[2].split('=')[1]
    write_dir = args[3].split('=')[1]
    source_name = args[4].split('=')[1]
    target_name = args[5].split('=')[1]
    split_seed = int(args[6].split('=')[1])
    holdout_frac = float(args[7].split('=')[1])
    # write params to dict
    params = {}
    params['data_fn'] = data_fn
    params['fold_fn'] = fold_fn
    params['write_dir'] = write_dir
    params['source_name'] = source_name
    params['target_name'] = target_name
    params['split_seed'] = split_seed
    params['holdout_frac'] = holdout_frac
    helpers.write_pickle(params, write_dir + '/params.pkl')
    return data_fn, fold_fn, write_dir, source_name, target_name, split_seed, holdout_frac

def predict_raw(source_df, source_col, target_test_sd):
    d = source_df.merge(target_test_sd, on=['sample_id', 'drug_id'], validate='one_to_one')
    assert len(d) == len(target_test_sd)
    predictions = d[source_col].to_numpy()
    return predictions

def evaluate_raw(predictions, target_test_df, target_col):
    test = target_test_df[target_col].to_numpy()
    return helpers.pearson_correlation(predictions, test)

def main():
    # get args
    data_fn, fold_fn, write_dir, source_name, target_name, split_seed, holdout_frac = get_raw_args(sys.argv, 7)
    source_col = 'log_' + source_name + '_published_auc_mean'
    target_col = 'log_' + target_name + '_published_auc_mean'
    source_df, target_train_df, target_test_df = helpers.get_source_and_target(data_fn, fold_fn, source_col, target_col, split_seed, holdout_frac)
    target_train_sd = helpers.get_sample_drug_ids(target_train_df)
    target_test_sd = helpers.get_sample_drug_ids(target_test_df)
    # TODO: get test pairs from target_test_df
    # in other files: 
    # get indices from source_df, target_train_df as well
    # zscore source_df and target_train_df
    # fit model
    # predict test results (note, these are still z-scored)
    #### using zscore mean from target_train_df, "inverse z_score" these predictions
    #### this should output a vector of test predictions
    # make predictions
    train_predictions = predict_raw(source_df, source_col, target_train_sd)
    test_predictions = predict_raw(source_df, source_col, target_test_sd)
    # evaluate predictions
    train_corr = evaluate_raw(train_predictions, target_train_df, target_col)
    test_corr = evaluate_raw(test_predictions, target_test_df, target_col)
    # save to file
    helpers.write_pickle(train_corr, write_dir + '/train.pkl')
    helpers.write_pickle(test_corr, write_dir + '/test.pkl')
    print('train_corr: ' + str(train_corr))
    print('test_corr: ' + str(test_corr))

if __name__ == "__main__":
    main()