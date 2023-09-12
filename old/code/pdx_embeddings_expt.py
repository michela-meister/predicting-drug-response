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
import embeddings_expt as embed
import cross_val
import helpers
import model_helpers as modeling

N_MODELS = 5

def save_params(data_fn, write_dir, n_steps):
    params = {}
    params['data_fn'] = data_fn
    params['write_dir'] = write_dir
    params['n_steps'] = n_steps
    helpers.write_pickle(params, write_dir + '/params.pkl')

def get_args(args, n):
    if len(args) != n + 1:
        print('Error! Expected ' + str(n + 1) + ' arguments but got ' + str(len(args)))
        return
    data_fn = args[1].split("=")[1]
    write_dir = args[2].split("=")[1]
    n_steps = int(args[3].split("=")[1])
    # verify that params are valid for method given and save
    assert n_steps > 0
    save_params(data_fn, write_dir, n_steps)
    return data_fn, write_dir, n_steps

def main():
    source_col = 'log10_ic50_(uM)'
    target_col = 'T_C'
    split_type = 'sample_split'
    holdout_frac = -1
    method = 'transfer'
    data_fn, write_dir, n_steps = get_args(sys.argv, 3)
    # Split target dataset into train and test, get number of samples and drugs in dataset
    target_train_df, n_samp, n_drug = embed.get_target_train_df(data_fn, target_col)
    # ================================
    # MAKE PREDICTIONS VIA TRANSFER
    source_df = helpers.get_source(data_fn, source_col)
    k = expt.choose_k('transfer', target_train_df, target_col, split_type, n_samp, n_drug, n_steps, source_df, source_col)
    # Run with chosen k to get embeddings
    train_predict_list, s_list, s_prime_list = embed.predict_embed_wrapper(source_df, source_col, target_train_df, target_col, n_samp, n_drug, n_steps, k)
    train_result, s, s_prime = embed.embed_evaluate(train_predict_list, s_list, s_prime_list, target_train_df, target_col)
    # SAVE EMBEDDINGS
    helpers.write_pickle(s, write_dir + '/s_embed.pkl')
    helpers.write_pickle(s_prime, write_dir + '/s_prime_embed.pkl')
    helpers.write_pickle(train_result, write_dir + '/train_result.pkl')



if __name__ == "__main__":
    main()