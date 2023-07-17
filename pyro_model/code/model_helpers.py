import logging
import os
import sys

import torch
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

import pyro
import graphviz
import pyro.distributions as dist
import pyro.distributions.constraints as constraints

import helpers
    
def get_model_inputs(train_fn, sample_fn, drug_fn):
    df = pd.read_pickle(train_fn)
    sample_dict = helpers.read_pickle(sample_fn)
    drug_dict = helpers.read_pickle(drug_fn)
    n_samp = len(sample_dict.keys())
    n_drug = len(drug_dict.keys())
    s_idx = df['s_idx'].to_numpy()
    d_idx = df['d_idx'].to_numpy()
    obs = torch.Tensor(df['log(V_V0)'])
    return n_samp, n_drug, s_idx, d_idx, obs

def transfer_model(n_samp, n_drug, s_idx1, d_idx1, s_idx2, d_idx2, params, obs1=None, n_obs1=None, obs2=None, n_obs2=None, k=1, r=1):
    print('TRANSFER!')
    print('K = ' + str(k))
    if obs1 is None and n_obs1 is None:
        print('Error!: both obs1 and n_obs1 are None.')
    if obs1 is not None:
        n_obs1 = obs1.shape[0]
    if obs2 is None and n_obs2 is None:
        print('Error: both obs2 and n_obs2 are None')
    if obs2 is not None:
        n_obs2 = obs2.shape[0]
    # create global offset
    a1_sigma = pyro.param('a1_sigma', dist.Gamma(params['alpha'], params['beta']), constraint=constraints.positive)
    a1 = pyro.sample('a1', dist.Normal(0, a1_sigma))   
    # create s
    s_sigma = pyro.param('s_sigma', dist.Gamma(params['alpha'], params['beta']), constraint=constraints.positive)
    with pyro.plate('s_plate', n_samp):
        with pyro.plate('k_s', k):
            s = pyro.sample('s', dist.Normal(0, s_sigma))
    s = torch.transpose(s, 0, 1)
    # create d
    d_sigma = pyro.param('d_sigma', dist.Gamma(params['alpha'], params['beta']), constraint=constraints.positive)
    with pyro.plate('d_plate', n_drug):
        with pyro.plate('k_d', k):
            d = pyro.sample('d', dist.Normal(0, d_sigma))
    # mat1 = sd
    mat1 = torch.matmul(s, d) # should be: n-samp x n-drug
    assert (mat1.shape[0] == n_samp) and (mat1.shape[1] == n_drug)
    mean1 = mat1[s_idx1, d_idx1] + a1
    sigma1 = pyro.sample('sigma1', dist.Gamma(params['alpha'], params['beta']))
    with pyro.plate('data1_plate', n_obs1):
        data1 = pyro.sample('data1', dist.Normal(mean1, sigma1 * torch.ones(n_obs1)), obs=obs1)
    # create global offset
    a2_sigma = pyro.param('a2_sigma', dist.Gamma(params['alpha'], params['beta']), constraint=constraints.positive)
    a2 = pyro.sample('a2', dist.Normal(0, a2_sigma))  
    # create W
    w_sigma = pyro.param('w_sigma', dist.Gamma(params['alpha'], params['beta']), constraint=constraints.positive)
    with pyro.plate('w_row_plate', n_samp):
        with pyro.plate('r_row', r):
            w_row = pyro.sample('w_row', dist.Normal(0, w_sigma))
    with pyro.plate('w_col_plate', n_samp):
        with pyro.plate('r_col', r):
            w_col = pyro.sample('w_col', dist.Normal(0, w_sigma))
    w_col = torch.transpose(w_col, 0, 1)
    W = torch.matmul(w_col, w_row)
    assert (W.shape[0] == n_samp) and (W.shape[1] == n_samp)
    # s' = Ws
    spr = torch.matmul(W, s)
    assert (spr.shape[0] == n_samp) and (spr.shape[1] == k)
    mat2 = torch.matmul(spr, d)
    assert (mat2.shape[0] == n_samp) and (mat2.shape[1] == n_drug)
    mean2 = mat2[s_idx2, d_idx2] + a2
    sigma2 = pyro.sample('sigma2', dist.Gamma(params['alpha'], params['beta']))
    with pyro.plate('data2_plate', n_obs2):
        data2 = pyro.sample('data2', dist.Normal(mean2, sigma2 * torch.ones(n_obs2)), obs=obs2)
    # DO: create different vecs a, vecs sigma

def vectorized_model(n_samp, n_drug, s_idx, d_idx, params, obs=None, n_obs=None, k=1):
    print('VECTORIZED!')
    print('K = ' + str(k))
    if obs is None and n_obs is None:
        print('Error!: both obs and n_obs are None.')
    if obs is not None:
        n_obs = obs.shape[0]
    # create global offset
    a_sigma = pyro.param('a_sigma', dist.Gamma(params['alpha'], params['beta']), constraint=constraints.positive)
    a = pyro.sample('a', dist.Normal(0, a_sigma))   
    # create s
    s_sigma = pyro.param('s_sigma', dist.Gamma(params['alpha'], params['beta']), constraint=constraints.positive)
    with pyro.plate('s_plate', n_samp):
        with pyro.plate('k1', k):
            s = pyro.sample('s', dist.Normal(0, s_sigma))
    # create d
    d_sigma = pyro.param('d_sigma', dist.Gamma(params['alpha'], params['beta']), constraint=constraints.positive)
    with pyro.plate('d_plate', n_drug):
        with pyro.plate('k2', k):
            d = pyro.sample('d', dist.Normal(0, d_sigma))
    # create data
    # rank-k matrix
    s = torch.transpose(s, 0, 1)
    mat = torch.matmul(s, d) # should be: n-samp x n-drug
    assert (mat.shape[0] == n_samp) and (mat.shape[1] == n_drug)
    mean = mat[s_idx, d_idx] + a
    sigma = pyro.sample('sigma', dist.Gamma(params['alpha'], params['beta']))
    with pyro.plate('data_plate', n_obs):
        data = pyro.sample('data', dist.Normal(mean, sigma * torch.ones(n_obs)), obs=obs)
    return data


# n_samp: number of samples
# n_drug: number of drugs
# obs: torch.Tensor of observations
# s_idx: numpy array where s_idx[i] is the index of the sample for the i-th observation
# d_idx: numpy array where d_idx[i] is the index of the drug for the i-th observation
def model(n_samp, n_drug, s_idx, d_idx, params, obs=None, n_obs=None, k=1):
    if k != 1:
        print('need k = 1!')
    print('NORMAL MODEL!')
    if obs is None and n_obs is None:
        print('Error!: both obs and n_obs are None.')
    if obs is not None:
        n_obs = obs.shape[0]
    # create global offset
    alpha = torch.Tensor([params['alpha']])
    beta = torch.Tensor([params['beta']])
    a_sigma = pyro.param('a_sigma', dist.Gamma(alpha, beta), constraint=constraints.positive)
    a = pyro.sample('a', dist.Normal(torch.zeros(()), a_sigma * torch.ones(())))   
    # create s
    s_sigma = pyro.param('s_sigma', dist.Gamma(alpha, beta), constraint=constraints.positive)
    a_s_sigma = pyro.param('a_s_sigma', dist.Gamma(alpha, beta), constraint=constraints.positive)
    with pyro.plate('s_plate', n_samp):
        a_s = pyro.sample('a_s', dist.Normal(torch.zeros(n_samp), a_s_sigma * torch.ones(n_samp)))
        s = pyro.sample('s', dist.Normal(torch.zeros(n_samp), s_sigma * torch.ones(n_samp)))
    # create d
    d_sigma = pyro.param('d_sigma', dist.Gamma(alpha, beta), constraint=constraints.positive)
    a_d_sigma = pyro.param('a_d_sigma', dist.Gamma(alpha, beta), constraint=constraints.positive)
    with pyro.plate('d_plate', n_drug):
        a_d = pyro.sample('a_d', dist.Normal(torch.zeros(n_drug), a_d_sigma * torch.ones(n_drug)))
        d = pyro.sample('d', dist.Normal(torch.zeros(n_drug), d_sigma * torch.ones(n_drug)))
    # create data
    mean = s[s_idx] * d[d_idx] + a_s[s_idx] + a_d[d_idx] + a
    sigma = pyro.sample('sigma', dist.Gamma(params['alpha'], params['beta']))
    with pyro.plate('data_plate', n_obs):
        data = pyro.sample('data', dist.Normal(mean, sigma * torch.ones(n_obs)), obs=obs)
    return data
