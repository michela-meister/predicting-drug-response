import numpy as np
import pandas as pd
import pyro.util
import sys

import helpers

def save_params(method, source, target, holdout_frac, data_fn, write_dir, fold_fn, hyp_fn, split_seed, model_seed, k, r):
    params = {}
    params['method'] = method
    params['source'] = source
    params['target'] = target
    params['holdout_frac'] = holdout_frac
    params['data_fn'] = data_fn
    params['write_dir'] = write_dir
    params['fold_fn'] = fold_fn
    params['hyp_fn'] = hyp_fn
    params['split_seed'] = split_seed
    params['model_seed'] = model_seed
    params['k'] = k
    params['r'] = r
    helpers.write_pickle(params, write_dir + '/params.pkl')

def check_params(method, source, target, holdout_frac, fold_fn, hyp_fn, split_seed, model_seed, k, r):
    assert method in ['raw', 'transfer', 'target_only']
    assert target in ['REP', 'GDSC', 'CTD2']
    assert (fold_fn != "") or (0 <= holdout_frac and holdout_frac <= 1)
    assert split_seed >= 0
    if method == 'raw':
        return
    assert model_seed >= 0
    assert hyp_fn != "" or k >= 0
    if method == 'target_only':
        return
    assert source in ['REP', 'GDSC', 'CTD2']
    assert hyp_fn != "" or r >= 0

def get_raw_args(args, n):
    if len(args) != n + 1:
        print('Error! Expected ' + str(n + 1) + ' arguments but got ' + str(len(args)))
        return
    method = args[1].split("=")[1]
    source = args[2].split("=")[1]
    target = args[3].split("=")[1]
    holdout_frac = float(args[4].split("=")[1])
    data_fn = args[5].split("=")[1]
    write_dir = args[6].split("=")[1]
    fold_fn = args[7].split("=")[1]
    hyp_fn = args[8].split("=")[1]
    split_seed = int(args[9].split("=")[1])
    model_seed = int(args[10].split("=")[1])
    k = int(args[11].split("=")[1])
    r = int(args[12].split("=")[1])
    # verify that params are valid for method given and save
    check_params(method, source, target, holdout_frac, fold_fn, hyp_fn, split_seed, model_seed, k, r)
    save_params(method, source, target, holdout_frac, data_fn, write_dir, fold_fn, hyp_fn, split_seed, model_seed, k, r)
    return method, source, target, holdout_frac, data_fn, write_dir, fold_fn, hyp_fn, split_seed, model_seed, k, r

def predict_raw_helper(source_df, source_col, target_sd):
    d = source_df.merge(target_sd, on=['sample_id', 'drug_id'], validate='one_to_one')
    assert len(d) == len(target_sd)
    predictions = d[source_col].to_numpy()
    return predictions

def predict_raw(source_df, source_col, target_train_sd, target_test_sd):
    train_predict = predict_raw_helper(source_df, source_col, target_train_sd)
    test_predict = predict_raw_helper(source_df, source_col, target_test_sd)
    return train_predict, test_predict

def predict_transfer(source_df, source_col, target_train_df, target_col, target_train_sd, target_test_sd):
    # get model inputs --> return indices, obs vector
    # zscore data --> return target_train params
    # fit model --> return model_predictions
    # inverse-zscore model_predictions --> return predictions
    print('Run Transfer!')
    train_predict = np.random.randn(len(target_train_sd))
    test_predict = np.random.randn(len(target_test_sd))
    return train_predict, test_predict

def predict_target_only(target_train_df, target_col, target_train_sd, target_test_sd):
    # get model inputs --> return indices, obs vector
    # zscore data --> return target_train params
    # fit model --> return model_predictions
    # inverse-zscore model_predictions --> return predictions
    print('Run Target Only!')
    train_predict = np.random.randn(len(target_train_sd))
    test_predict = np.random.randn(len(target_test_sd))
    return train_predict, test_predict

def evaluate(predictions, target_test_df, target_col):
    test = target_test_df[target_col].to_numpy()
    return helpers.pearson_correlation(predictions, test)

# Returns column names in dataset for given method
def get_column_names(method, source_name, target_name):
    suffix = '_published_auc_mean'
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
    method, source_name, target_name, holdout_frac, data_fn, write_dir, fold_fn, hyp_fn, split_seed, model_seed, k, r = get_raw_args(sys.argv, 12)
    source_col, target_col = get_column_names(method, source_name, target_name)
    # Split dataset, get sample_ids and drug_ids for target
    source_df, target_train_df, target_test_df = helpers.get_source_and_target(data_fn, fold_fn, source_col, target_col, split_seed, holdout_frac)
    target_train_sd = helpers.get_sample_drug_ids(target_train_df)
    target_test_sd = helpers.get_sample_drug_ids(target_test_df)
    # ================================
    # MAKE PREDICTIONS BY METHOD
    if method == 'raw':
        train_predict, test_predict = predict_raw(source_df, source_col, target_train_sd, target_test_sd)
    elif method == 'transfer':
        train_predict, test_predict = predict_transfer(source_df, source_col, target_train_df, target_col, target_train_sd, target_test_sd)
    elif method == 'target_only':
        train_predict, test_predict = predict_target_only(target_train_df, target_col, target_train_sd, target_test_sd)
    else:
        print('Error! Method must be one of: raw, transfer, target_only.')
        return
    # get model inputs
    #### get indices
    #### get obs vector
    # zscore data --> return params
    # fit model
    # make predictions (take in zscore data for target_train)

    # in other files: 
    # get indices from source_df, target_train_df as well
    # zscore source_df and target_train_df
    # fit model
    # predict test results (note, these are still z-scored)
    #### using zscore mean from target_train_df, "inverse z_score" these predictions
    #### this should output a vector of test predictions
    # ================================
    # evaluate predictions
    train_corr = evaluate(train_predict, target_train_df, target_col)
    test_corr = evaluate(test_predict, target_test_df, target_col)
    # save to file
    helpers.write_pickle(train_corr, write_dir + '/train.pkl')
    helpers.write_pickle(test_corr, write_dir + '/test.pkl')
    print('train_corr: ' + str(train_corr))
    print('test_corr: ' + str(test_corr))

if __name__ == "__main__":
    main()