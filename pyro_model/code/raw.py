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

# Real values
#K_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
#N_MODELS = 10
#N_SPLITS = 10

# Test Values
K_LIST = [5, 10]
N_MODELS = 2
N_SPLITS = 2

def save_params(method, source, target, holdout_frac, data_fn, write_dir, fold_fn, hyp_fn, split_seed, model_seed, k, r, n_steps, split_type):
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
    params['n_steps'] = n_steps
    params['split_type'] = split_type
    helpers.write_pickle(params, write_dir + '/params.pkl')

def check_params(method, source, target, holdout_frac, fold_fn, hyp_fn, split_seed, model_seed, k, r, n_steps, split_type):
    assert method in ['raw', 'transfer', 'target_only']
    assert target in ['REP', 'GDSC', 'CTD2', 'synth']
    assert split_seed >= 0
    if method == 'raw':
        # raw can handle sample folds or random pairs
        assert (fold_fn != "") or (0 <= holdout_frac and holdout_frac <= 1)
        return
    assert split_type in ['random_split', 'sample_split']
    assert n_steps > 0
    assert model_seed >= 0
    assert hyp_fn != "" or k >= 0
    if method == 'target_only':
        # target_only can't handle sample folds
        assert split_type == 'random_split' 
        assert 0 <= holdout_frac and holdout_frac <= 1
        return
    assert source in ['REP', 'GDSC', 'CTD2']
    # transfer method can handle sample folds or random pairs
    assert (split_type == 'sample_split' and fold_fn != "") or ((split_type == 'random_split') and (0 <= holdout_frac and holdout_frac <= 1))
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
    n_steps = int(args[13].split("=")[1])
    # get split type
    if fold_fn == "":
        split_type = 'random_split'
    else:
        split_type = 'sample_split'
    # verify that params are valid for method given and save
    check_params(method, source, target, holdout_frac, fold_fn, hyp_fn, split_seed, model_seed, k, r, n_steps, split_type)
    save_params(method, source, target, holdout_frac, data_fn, write_dir, fold_fn, hyp_fn, split_seed, model_seed, k, r, n_steps, split_type)
    return method, source, target, holdout_frac, data_fn, write_dir, fold_fn, hyp_fn, split_seed, model_seed, k, r, n_steps, split_type

def predict_raw_helper(source_df, source_col, target_sd):
    d = source_df.merge(target_sd, on=['sample_id', 'drug_id'], validate='one_to_one')
    assert len(d) == len(target_sd)
    predictions = d[source_col].to_numpy()
    return predictions

def predict_raw(source_df, source_col, target_train_sd, target_test_sd):
    train_predict = predict_raw_helper(source_df, source_col, target_train_sd)
    test_predict = predict_raw_helper(source_df, source_col, target_test_sd)
    return train_predict, test_predict

def matrix_transfer(s, d, w_row, w_col, k, r):
    W = np.matmul(np.transpose(w_col), w_row)
    # s already comes transposed, as defined in model
    assert s.shape[0] == k
    assert W.shape[0] == k and W.shape[0] == k
    s_prime = np.matmul(W, s) 
    mat2 = np.matmul(np.transpose(s_prime), d)
    return mat2

def predict_transfer(model_seed, s_idx_src, d_idx_src, obs_src, s_idx_train, d_idx_train, obs_train, s_idx_test, d_idx_test, n_samp, n_drug, n_steps, k, r):
    print('Transfer, k: ' + str(k) + ', r: ' + str(r))
    # FIT MODEL
    pyro.clear_param_store()
    pyro.util.set_rng_seed(model_seed)
    adam_params = {"lr": 0.05}
    optimizer = Adam(adam_params)
    autoguide = AutoNormal(modeling.transfer_model)
    svi = SVI(modeling.transfer_model, autoguide, optimizer, loss=Trace_ELBO())
    losses = []
    for step in tqdm.trange(n_steps):
        svi.step(n_samp, n_drug, s_idx_src, d_idx_src, s_idx_train, d_idx_train, obs_src, len(obs_src), obs_train, len(obs_train), r=r, k=k)
        loss = svi.evaluate_loss(n_samp, n_drug, s_idx_src, d_idx_src, s_idx_train, d_idx_train, obs_src, len(obs_src), obs_train, len(obs_train), r=r, k=k)
        losses.append(loss)
    # MAKE INITIAL PREDICTIONS BASED ON MODEL
    # retrieve values out for s and d vectors
    s_loc = pyro.param("AutoNormal.locs.s").detach().numpy()
    d_loc = pyro.param("AutoNormal.locs.d").detach().numpy()
    # need to retrive w_col, w_row and reconstruct s'!
    w_row_loc = pyro.param("AutoNormal.locs.w_row").detach().numpy()
    w_col_loc = pyro.param("AutoNormal.locs.w_col").detach().numpy()
    # predict function: takes in w_row, w_col, s, d --> mat2
    mat = matrix_transfer(s_loc, d_loc, w_row_loc, w_col_loc, k, r)
    train_means = mat[s_idx_train, d_idx_train]
    test_means = mat[s_idx_test, d_idx_test]
    return train_means, test_means

def run_predict_transfer(model_seed, source_df, source_col, target_train_df, target_col, s_idx_test, d_idx_test, n_samp, n_drug, n_steps, k):
    s_idx_train, d_idx_train = helpers.get_sample_drug_indices(target_train_df)
    obs_train = target_train_df[target_col].to_numpy()
    mu, sigma, obs_train = helpers.zscore(obs_train)
    obs_train = torch.Tensor(obs_train)
    s_idx_src, d_idx_src = helpers.get_sample_drug_indices(source_df)
    obs_src = source_df[source_col].to_numpy()
    _, _, obs_src = helpers.zscore(obs_src)
    obs_src = torch.Tensor(obs_src)
    train_initial, test_initial = predict_transfer(model_seed, s_idx_src, d_idx_src, obs_src, s_idx_train, d_idx_train, obs_train, s_idx_test, 
        d_idx_test, n_samp, n_drug, n_steps, k, k)
    train_predict = helpers.inverse_zscore(train_initial, mu, sigma)
    test_predict = helpers.inverse_zscore(test_initial, mu, sigma)
    assert len(train_predict) == len(s_idx_train)
    assert len(test_predict) == len(s_idx_test)
    return train_predict, test_predict

def predict_transfer_wrapper(source_df, source_col, target_train_df, target_col, s_idx_test, d_idx_test, n_samp, n_drug, n_steps, k):
    train_predict_list = []
    test_predict_list = []
    for model_seed in range(0, N_MODELS):
        train_predict, test_predict = run_predict_transfer(model_seed, source_df, source_col, target_train_df, target_col, s_idx_test, d_idx_test, n_samp, 
            n_drug, n_steps, k)
        train_predict_list.append(train_predict)
        test_predict_list.append(test_predict)
    return train_predict_list, test_predict_list

def matrix_target_only(s, d, k):
    assert s.shape[0] == k
    assert d.shape[0] == k
    return np.matmul(np.transpose(s), d)

def predict_target_only(model_seed, s_idx_train, d_idx_train, obs_train, s_idx_test, d_idx_test, n_samp, n_drug, n_steps, k):
    print('Target only, k: ' + str(k))
    # FIT MODEL
    pyro.clear_param_store()
    pyro.util.set_rng_seed(model_seed)
    adam_params = {"lr": 0.05}
    optimizer = Adam(adam_params)
    autoguide = AutoNormal(modeling.target_only_model)
    svi = SVI(modeling.target_only_model, autoguide, optimizer, loss=Trace_ELBO())
    losses = []
    for step in tqdm.trange(n_steps):
        svi.step(n_samp, n_drug, s_idx_train, d_idx_train, obs_train, len(obs_train), k=k)
        loss = svi.evaluate_loss(n_samp, n_drug, s_idx_train, d_idx_train, obs_train, len(obs_train), k=k)
        losses.append(loss)
    # MAKE INITIAL PREDICTIONS BASED ON MODEL
    # retrieve values out for s and d vectors
    s_loc = pyro.param("AutoNormal.locs.s").detach().numpy()
    d_loc = pyro.param("AutoNormal.locs.d").detach().numpy()
    mat = matrix_target_only(s_loc, d_loc, k)
    train_means = mat[s_idx_train, d_idx_train]
    test_means = mat[s_idx_test, d_idx_test]
    return train_means, test_means

def run_predict_target_only(model_seed, target_train_df, target_col, s_idx_test, d_idx_test, n_samp, n_drug, n_steps, k):
    s_idx_train, d_idx_train = helpers.get_sample_drug_indices(target_train_df)
    obs_train = target_train_df[target_col].to_numpy()
    mu, sigma, obs_train = helpers.zscore(obs_train)
    obs_train = torch.Tensor(obs_train)
    train_initial, test_initial = predict_target_only(model_seed, s_idx_train, d_idx_train, obs_train, s_idx_test, d_idx_test, n_samp, n_drug, n_steps, k)
    train_predict = helpers.inverse_zscore(train_initial, mu, sigma)
    test_predict = helpers.inverse_zscore(test_initial, mu, sigma)
    assert len(train_predict) == len(s_idx_train)
    assert len(test_predict) == len(s_idx_test)
    return train_predict, test_predict

def predict_target_only_wrapper(target_train_df, target_col, s_idx_test, d_idx_test, n_samp, n_drug, n_steps, k):
    train_predict_list = []
    test_predict_list = []
    for model_seed in range(0, N_MODELS):
        train_predict, test_predict = run_predict_target_only(model_seed, target_train_df, target_col, s_idx_test, d_idx_test, n_samp, n_drug, n_steps, k)
        train_predict_list.append(train_predict)
        test_predict_list.append(test_predict)
    return train_predict_list, test_predict_list

def evaluate_correlation(predictions, df, col):
    test = df[col].to_numpy()
    return helpers.pearson_correlation(predictions, test)

def evaluate(train_predict_list, test_predict_list, target_train_df, target_test_df, target_col):
    assert len(train_predict_list) == len(test_predict_list)
    n_models = len(train_predict_list)
    train_corr_list = []
    test_corr_list = []
    for i in range(0, N_MODELS):
        train_corr = evaluate_correlation(train_predict_list[i], target_train_df, target_col)
        test_corr = evaluate_correlation(test_predict_list[i], target_test_df, target_col)
        train_corr_list.append(train_corr)
        test_corr_list.append(test_corr)
    idx = np.argmax(train_corr_list)
    train_result = train_corr_list[idx]
    test_result = test_corr_list[idx]
    train_predictions = train_predict_list[idx]
    test_predictions = test_predict_list[idx]
    return train_result, test_result, train_predictions, test_predictions

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

def obs_to_tensor(vec):
    return torch.Tensor(vec)

def choose_k(method, target_train_df, target_col, split_type, n_samp, n_drug, n_steps, source_df=None, source_col=None):
    print('Choose k via cross validation')
    assert method in ['target_only', 'transfer']
    assert split_type in ['random_split', 'sample_split']
    if method == 'target_only':
        assert split_type == 'random_split'
    if method == 'transfer':
        assert source_df is not None
        assert source_col is not None
    # get data (either pairs or samples) based on split_type
    X = cross_val.get_items_to_split(target_train_df, split_type)
    opt_k_list = []
    kf = KFold(n_splits=N_SPLITS, random_state=707, shuffle=True)
    kf.get_n_splits(X)
    for i, (train_index, val_index) in enumerate(kf.split(X)):
        print("Fold: " + str(i + 1))
        train_df, val_df = cross_val.split_dataframe(target_train_df, 'sample_id', 'drug_id', X, split_type, train_index, val_index)
        train_corr_list = []
        val_corr_list = []
        for k in K_LIST:
            s_idx_val, d_idx_val = helpers.get_sample_drug_indices(val_df)
            print('Run ' + str(N_MODELS) + ' model restarts')
            if method == 'target_only':
                train_predict_list, val_predict_list = predict_target_only_wrapper(train_df, target_col, s_idx_val, d_idx_val, n_samp, n_drug, n_steps, k)
            elif method == 'transfer':
                train_predict_list, val_predict_list = predict_transfer_wrapper(source_df, source_col, train_df, target_col, s_idx_val, d_idx_val, n_samp, 
                    n_drug, n_steps, k)
            train_corr, val_corr, _, _ = evaluate(train_predict_list, val_predict_list, train_df, val_df, target_col)
            train_corr_list.append(train_corr)
            val_corr_list.append(val_corr)
            print('train_corr: ' + str(train_corr))
            print('val_corr: ' + str(val_corr))
        # get the index associated with the highest correlation on the validation set
        opt_index = np.argmax(val_corr_list)
        # save the value of k corresponding to the above index
        opt_k = K_LIST[opt_index]
        print(opt_k)
        opt_k_list.append(opt_k)
    return int(np.mean(opt_k_list))

def save_predictions(write_fn, predictions, df):
    assert len(predictions) == len(df)
    d = {'predictions': predictions, 'sample_ids': df['sample_id'].to_numpy(), 'drug_id': df['drug_id'].to_numpy()}
    helpers.write_pickle(d, write_fn)

def main():
    method, source_name, target_name, holdout_frac, data_fn, write_dir, fold_fn, hyp_fn, split_seed, model_seed, k, r, n_steps, split_type = get_raw_args(sys.argv, 13)
    source_col, target_col = get_column_names(method, source_name, target_name)
    # Split target dataset into train and test, get number of samples and drugs in dataset
    target_train_df, target_test_df, n_samp, n_drug = helpers.get_target(data_fn, fold_fn, target_col, split_seed, holdout_frac, split_type)
    # ================================
    # MAKE PREDICTIONS BY METHOD
    assert method in ['raw', 'target_only', 'transfer']
    if method == 'raw':
        target_train_sd = helpers.get_sample_drug_ids(target_train_df)
        target_test_sd = helpers.get_sample_drug_ids(target_test_df)
        source_df = helpers.get_source(data_fn, source_col)
        train_predictions, test_predictions = predict_raw(source_df, source_col, target_train_sd, target_test_sd)
        train_corr = evaluate_correlation(train_predictions, target_train_df, target_col)
        test_corr = evaluate_correlation(test_predictions, target_test_df, target_col)
    else:
        s_idx_test, d_idx_test = helpers.get_sample_drug_indices(target_test_df)
        if method == 'target_only':
            k = choose_k('target_only', target_train_df, target_col, split_type, n_samp, n_drug, n_steps)
            train_predict_list, test_predict_list = predict_target_only_wrapper(target_train_df, target_col, s_idx_test, d_idx_test, n_samp, n_drug, n_steps, k)
        elif method == 'transfer':
            source_df = helpers.get_source(data_fn, source_col)
            k = choose_k('transfer', target_train_df, target_col, split_type, n_samp, n_drug, n_steps, source_df, source_col)
            train_predict_list, test_predict_list = predict_transfer_wrapper(source_df, source_col, target_train_df, target_col, s_idx_test, d_idx_test, n_samp, 
                n_drug, n_steps, k)
        train_corr, test_corr, train_predictions, test_predictions = evaluate(train_predict_list, test_predict_list, target_train_df, target_test_df, target_col)
    # ================================
    # EVALUATE PREDICTIONS AND SAVE
    # save to file
    helpers.write_pickle(train_corr, write_dir + '/train.pkl')
    helpers.write_pickle(test_corr, write_dir + '/test.pkl')
    save_predictions(write_dir + '/train_predictions.pkl', train_predictions, target_train_df)
    save_predictions(write_dir + '/test_predictions.pkl', test_predictions, target_test_df)
    # TODO: save predictions!!

    print('train_corr: ' + str(train_corr))
    print('test_corr: ' + str(test_corr))

if __name__ == "__main__":
    main()