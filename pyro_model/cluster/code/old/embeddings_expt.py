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

N_MODELS = 5

def get_s_prime(s, w_row, w_col, k):
    W = np.matmul(np.transpose(w_col), w_row)
    # s already comes transposed, as defined in model
    assert s.shape[0] == k
    assert W.shape[0] == k and W.shape[0] == k
    s_prime = np.matmul(W, s)
    return s_prime

def embed_transfer(model_seed, s_idx_src, d_idx_src, obs_src, s_idx_train, d_idx_train, obs_train, n_samp, n_drug, n_steps, k, r):
    #print('Transfer, k: ' + str(k) + ', r: ' + str(r))
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
    mat = expt.matrix_transfer(s_loc, d_loc, w_row_loc, w_col_loc, k, r)
    train_means = mat[s_idx_train, d_idx_train]
    s_prime = get_s_prime(s_loc, w_row_loc, w_col_loc, k)
    return train_means, s_loc, s_prime

def run_embed_transfer(model_seed, source_df, source_col, target_train_df, target_col, n_samp, n_drug, n_steps, k):
    s_idx_train, d_idx_train = helpers.get_sample_drug_indices(target_train_df)
    obs_train = target_train_df[target_col].to_numpy()
    mu, sigma, obs_train = helpers.zscore(obs_train)
    obs_train = torch.Tensor(obs_train)
    s_idx_src, d_idx_src = helpers.get_sample_drug_indices(source_df)
    obs_src = source_df[source_col].to_numpy()
    _, _, obs_src = helpers.zscore(obs_src)
    obs_src = torch.Tensor(obs_src)
    train_initial, s, s_prime = embed_transfer(model_seed, s_idx_src, d_idx_src, obs_src, s_idx_train, d_idx_train, obs_train,
        n_samp, n_drug, n_steps, k, k)
    train_predict = helpers.inverse_zscore(train_initial, mu, sigma)
    return train_predict, s, s_prime

def predict_embed_wrapper(source_df, source_col, target_train_df, target_col, n_samp, n_drug, n_steps, k):
    train_predict_list = []
    s_list = []
    s_prime_list = []
    for model_seed in range(0, N_MODELS):
        train_predict, s, s_prime = run_embed_transfer(model_seed, source_df, source_col, target_train_df, target_col, n_samp, n_drug, n_steps, k)
        train_predict_list.append(train_predict)
        s_list.append(s)
        s_prime_list.append(s_prime)
    return train_predict_list, s_list, s_prime_list

def embed_evaluate(train_predict_list, s_list, s_prime_list, target_train_df, target_col):
    assert len(train_predict_list) == len(s_list)
    assert len(train_predict_list) == len(s_prime_list)
    n_models = len(train_predict_list)
    train_corr_list = []
    for i in range(0, N_MODELS):
        train_corr = expt.evaluate_correlation(train_predict_list[i], target_train_df, target_col)
        train_corr_list.append(train_corr)
    idx = np.argmax(train_corr_list)
    train_result = train_corr_list[idx]
    s_result = s_list[idx]
    s_prime_result = s_prime_list[idx]
    return train_result, s_result, s_prime_result

def save_params(source, target, data_fn, write_dir, n_steps):
    params = {}
    params['source'] = source
    params['target'] = target
    params['data_fn'] = data_fn
    params['write_dir'] = write_dir
    params['n_steps'] = n_steps
    helpers.write_pickle(params, write_dir + '/params.pkl')

def get_args(args, n):
    if len(args) != n + 1:
        print('Error! Expected ' + str(n + 1) + ' arguments but got ' + str(len(args)))
        return
    source = args[1].split("=")[1]
    target = args[2].split("=")[1]
    data_fn = args[3].split("=")[1]
    write_dir = args[4].split("=")[1]
    n_steps = int(args[5].split("=")[1])
    # verify that params are valid for method given and save
    assert n_steps > 0
    save_params(source, target, data_fn, write_dir, n_steps)
    return source, target, data_fn, write_dir, n_steps

def get_target_train_df(data_fn, target_col):
    df = pd.read_csv(data_fn)
    assert len(df) == len(df.drop_duplicates())
    n_samp = df.sample_id.nunique()
    n_drug = df.drug_id.nunique()
    target_train_df = df[['sample_id', 'drug_id', target_col]]
    assert len(target_train_df) == len(df)
    return target_train_df, n_samp, n_drug

def main():
    method = 'transfer'
    split_type = 'sample_split'
    # TODO: Re-write args (and save!)
    source_name, target_name, data_fn, write_dir, n_steps = get_args(sys.argv, 5)
    source_col, target_col = expt.get_column_names(method, source_name, target_name)
    # Split target dataset into train and test, get number of samples and drugs in dataset
    target_train_df, n_samp, n_drug = get_target_train_df(data_fn, target_col)
    # ================================
    # MAKE PREDICTIONS VIA TRANSFER
    source_df = helpers.get_source(data_fn, source_col)
    k = expt.choose_k('transfer', target_train_df, target_col, split_type, n_samp, n_drug, n_steps, source_df, source_col)
    # Run with chosen k to get embeddings
    train_predict_list, s_list, s_prime_list = predict_embed_wrapper(source_df, source_col, target_train_df, target_col, n_samp, n_drug, n_steps, k)
    train_result, s, s_prime = embed_evaluate(train_predict_list, s_list, s_prime_list, target_train_df, target_col)
    # SAVE EMBEDDINGS
    helpers.write_pickle(s, write_dir + '/s_embed.pkl')
    helpers.write_pickle(s_prime, write_dir + '/s_prime_embed.pkl')
    helpers.write_pickle(train_result, write_dir + '/train_result.pkl')



if __name__ == "__main__":
    main()