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

def vectorized_model(n_samp, n_drug, s_idx, d_idx, params, obs=None, n_obs=None, k=1):
    print('VECTORIZED!')
    print('K = ' + str(k))
    if obs is None and n_obs is None:
        print('Error!: both obs and n_obs are None.')
    if obs is not None:
        n_obs = obs.shape[0]
    # create global offset
    a_sigma = pyro.param('a_sigma', dist.Gamma(params['alpha'], params['beta']), constraint=constraints.positive)
    a = pyro.sample('a', dist.Normal(torch.zeros(()), a_sigma * torch.ones(())))   
    # create s
    s_sigma = pyro.param('s_sigma', dist.Gamma(params['alpha'], params['beta']), constraint=constraints.positive)
    with pyro.plate('s_plate', n_samp):
        s = pyro.sample('s', dist.Normal(torch.zeros((k, n_samp)), s_sigma * torch.ones((k, n_samp))))
    # create d
    d_sigma = pyro.param('d_sigma', dist.Gamma(params['alpha'], params['beta']), constraint=constraints.positive)
    with pyro.plate('d_plate', n_drug):
        d = pyro.sample('d', dist.Normal(torch.zeros((k, n_drug)), d_sigma * torch.ones((k, n_drug))))
    # create data
    # rank-k matrix
    s = torch.transpose(s, 0, 1)
    mat = torch.matmul(s, d)
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
def model(n_samp, n_drug, s_idx, d_idx, params, obs=None, n_obs=None):
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
