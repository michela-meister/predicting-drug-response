import numpy as np
import pandas as pd
import pyro
import pyro.util
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
import sys
import torch
import tqdm

import helpers
import model_helpers as modeling

def save_params(method, source, target, holdout_frac, data_fn, write_dir, fold_fn, hyp_fn, split_seed, model_seed, k, r, n_steps):
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
    helpers.write_pickle(params, write_dir + '/params.pkl')

def check_params(method, source, target, holdout_frac, fold_fn, hyp_fn, split_seed, model_seed, k, r, n_steps):
    assert method in ['raw', 'transfer', 'target_only']
    assert target in ['REP', 'GDSC', 'CTD2', 'synth']
    assert split_seed >= 0
    if method == 'raw':
        # raw can handle sample folds or random pairs
        assert (fold_fn != "") or (0 <= holdout_frac and holdout_frac <= 1)
        return
    assert n_steps > 0
    assert model_seed >= 0
    assert hyp_fn != "" or k >= 0
    if method == 'target_only':
        # target_only can't handle sample folds
        assert fold_fn == "" 
        assert 0 <= holdout_frac and holdout_frac <= 1
        return
    assert source in ['REP', 'GDSC', 'CTD2']
    # transfer method can handle sample folds or random pairs
    assert (fold_fn != "") or (0 <= holdout_frac and holdout_frac <= 1)
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
    # verify that params are valid for method given and save
    check_params(method, source, target, holdout_frac, fold_fn, hyp_fn, split_seed, model_seed, k, r, n_steps)
    save_params(method, source, target, holdout_frac, data_fn, write_dir, fold_fn, hyp_fn, split_seed, model_seed, k, r, n_steps)
    return method, source, target, holdout_frac, data_fn, write_dir, fold_fn, hyp_fn, split_seed, model_seed, k, r, n_steps

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
    # FIT MODEL
    pyro.clear_param_store()
    pyro.util.set_rng_seed(model_seed)
    adam_params = {"lr": 0.05}
    optimizer = Adam(adam_params)
    autoguide = AutoNormal(modeling.transfer_model)
    svi = SVI(modeling.transfer_model, autoguide, optimizer, loss=Trace_ELBO())
    losses = []
    # TODO: Find / Replace!
    s_idx1 = s_idx_src
    d_idx1 = d_idx_src
    obs_1 = obs_src
    s_idx2 = s_idx_train
    d_idx2 = d_idx_train
    obs_2 = obs_train
    s_test_idx = s_idx_test
    d_test_idx = d_idx_test
    for step in tqdm.trange(n_steps):
        svi.step(n_samp, n_drug, s_idx1, d_idx1, s_idx2, d_idx2, obs_1, len(obs_1), obs_2, len(obs_2), r=r, k=k)
        loss = svi.evaluate_loss(n_samp, n_drug, s_idx1, d_idx1, s_idx2, d_idx2, obs_1, len(obs_1), obs_2, len(obs_2), r=r, k=k)
        losses.append(loss)
    print('FINAL LOSS DIFF: ' + str(losses[len(losses) - 1] - losses[len(losses) - 2]))
    # MAKE INITIAL PREDICTIONS BASED ON MODEL
    # retrieve values out for s and d vectors
    s_loc = pyro.param("AutoNormal.locs.s").detach().numpy()
    d_loc = pyro.param("AutoNormal.locs.d").detach().numpy()
    # need to retrive w_col, w_row and reconstruct s'!
    w_row_loc = pyro.param("AutoNormal.locs.w_row").detach().numpy()
    w_col_loc = pyro.param("AutoNormal.locs.w_col").detach().numpy()
    # predict function: takes in w_row, w_col, s, d --> mat2
    mat = matrix_transfer(s_loc, d_loc, w_row_loc, w_col_loc, k, r)
    train_means = mat[s_idx2, d_idx2]
    test_means = mat[s_test_idx, d_test_idx]
    return train_means, test_means

def matrix_target_only(s, d, k):
    assert s.shape[0] == k
    assert d.shape[0] == k
    return np.matmul(np.transpose(s), d)

def predict_target_only(model_seed, s_idx_train, d_idx_train, obs_train, s_idx_test, d_idx_test, n_samp, n_drug, n_steps, k):
    # FIT MODEL
    pyro.clear_param_store()
    pyro.util.set_rng_seed(model_seed)
    adam_params = {"lr": 0.05}
    optimizer = Adam(adam_params)
    autoguide = AutoNormal(modeling.target_only_model)
    svi = SVI(modeling.target_only_model, autoguide, optimizer, loss=Trace_ELBO())
    losses = []
    # TODO: Find / Replace!
    s_idx = s_idx_train
    d_idx = d_idx_train
    obs = obs_train
    s_test_idx = s_idx_test
    d_test_idx = d_idx_test
    for step in tqdm.trange(n_steps):
        # target_only_model(n_samp, n_drug, s_idx, d_idx, params, obs=None, n_obs=None, k=1)
        svi.step(n_samp, n_drug, s_idx, d_idx, obs, len(obs), k=k)
        loss = svi.evaluate_loss(n_samp, n_drug, s_idx, d_idx, obs, len(obs), k=k)
        losses.append(loss)
    print('FINAL LOSS DIFF: ' + str(losses[len(losses) - 1] - losses[len(losses) - 2]))
    # MAKE INITIAL PREDICTIONS BASED ON MODEL
    # retrieve values out for s and d vectors
    s_loc = pyro.param("AutoNormal.locs.s").detach().numpy()
    d_loc = pyro.param("AutoNormal.locs.d").detach().numpy()
    mat = matrix_target_only(s_loc, d_loc, k)
    train_means = mat[s_idx, d_idx]
    test_means = mat[s_test_idx, d_test_idx]
    return train_means, test_means

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

def obs_to_tensor(vec):
    return torch.Tensor(vec)

#def choose_k(train_df, method, split_type):
    # split train_df into folds
    # k_list = []
    # for each fold:
        # val = fold
        # train = other folds
        # result_list = []
        # for each value of k:
            # fit model using train, k, method
            # evaluate model on val
            # append val_corr to result_list
        # opt_k = the value of k associated with the highest result in result_list
        # k_list.append(opt_k)
    # return average(k_list)

# def predict_target_only():
#     k_best = choose_k_target_only(target_train_df)
#     train_means, test_means = fit_target_only_model()
#     return train_means, test_means

# def main():
    # get raw args
    # get column names
    # train / test split
    # k_best <---- choose_k: do 10-fold cross validation on train (iterating over every 5 k's)
    # fit train using k_best
    # evaluate on test
    # save predictions

def main():
    method, source_name, target_name, holdout_frac, data_fn, write_dir, fold_fn, hyp_fn, split_seed, model_seed, k, r, n_steps = get_raw_args(sys.argv, 13)
    source_col, target_col = get_column_names(method, source_name, target_name)
    # Split dataset, get sample_ids and drug_ids for target
    target_train_df, target_test_df, n_samp, n_drug = helpers.get_target(data_fn, fold_fn, target_col, split_seed, holdout_frac)
    target_train_sd = helpers.get_sample_drug_ids(target_train_df)
    target_test_sd = helpers.get_sample_drug_ids(target_test_df)
    # ================================
    # MAKE PREDICTIONS BY METHOD
    if method == 'raw':
        source_df = helpers.get_source(data_fn, fold_fn, source_col)
        train_predict, test_predict = predict_raw(source_df, source_col, target_train_sd, target_test_sd)
    else:
        # get target model inputs
        s_idx_train, d_idx_train = helpers.get_sample_drug_indices(target_train_df)
        s_idx_test, d_idx_test = helpers.get_sample_drug_indices(target_test_df)
        obs_train = target_train_df[target_col].to_numpy()
        mu, sigma, obs_train = helpers.zscore(obs_train)
        obs_train = torch.Tensor(obs_train)
        if method == 'target_only':
            train_initial, test_initial = predict_target_only(model_seed, s_idx_train, d_idx_train, obs_train, s_idx_test, d_idx_test, n_samp, n_drug, n_steps, k)
        elif method == 'transfer':
            # get source model inputs
            source_df = helpers.get_source(data_fn, fold_fn, source_col)
            s_idx_src, d_idx_src = helpers.get_sample_drug_indices(source_df)
            obs_src = source_df[source_col].to_numpy()
            _, _, obs_src = helpers.zscore(obs_src)
            obs_src = torch.Tensor(obs_src)
            # zscore source data
            train_initial, test_initial = predict_transfer(model_seed, s_idx_src, d_idx_src, obs_src, s_idx_train, d_idx_train, obs_train, s_idx_test, 
                d_idx_test, n_samp, n_drug, n_steps, k, r)
        else:
            print('Error! Method must be one of: raw, transfer, target_only.')
            return
        # invert zscore on both target_train and target_test predictions using target_train params
        train_predict = helpers.inverse_zscore(train_initial, mu, sigma)
        test_predict = helpers.inverse_zscore(test_initial, mu, sigma)
    assert len(train_predict) == len(target_train_df)
    assert len(test_predict) == len(target_test_df)
    # ================================
    # EVALUATE PREDICTIONS AND SAVE
    train_corr = evaluate(train_predict, target_train_df, target_col)
    test_corr = evaluate(test_predict, target_test_df, target_col)
    # save to file
    helpers.write_pickle(train_corr, write_dir + '/train.pkl')
    helpers.write_pickle(test_corr, write_dir + '/test.pkl')
    print('train_corr: ' + str(train_corr))
    print('test_corr: ' + str(test_corr))

if __name__ == "__main__":
    main()