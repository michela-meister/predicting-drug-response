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

# Parameters for Gamma distributions
ALPHA = .01
BETA = .01

# BMT transfer model.
# n_samp: number of samples total
# n_drug: number of drugs total
# s_idx1: list of sample source indices 
# d_idx1: list of drug source indices 
# s_idx2: list of sample target traning indices 
# d_idx2: list of drug target training indices 
# obs1: list of source observations 
# n_obs1: length of obs1
# obs2: list of target training observationt 
# n_obs2: length of obs2
# k: dimension of latent vectors and W matrix 
# r: deprecated; outer functions set equal to k
# Note: in the paper a sample has latent vectors c and c'; here they are called s and s'.
def transfer_model(n_samp, n_drug, s_idx1, d_idx1, s_idx2, d_idx2, obs1=None, n_obs1=None, obs2=None, n_obs2=None, k=1, r=1):
    assert r <= k
    if obs1 is None and n_obs1 is None:
        print('Error!: both obs1 and n_obs1 are None.')
    if obs1 is not None:
        n_obs1 = obs1.shape[0]
    if obs2 is None and n_obs2 is None:
        print('Error: both obs2 and n_obs2 are None')
    if obs2 is not None:
        n_obs2 = obs2.shape[0]
    # create global offset
    a1_sigma = pyro.sample('a1_sigma', dist.Gamma(ALPHA, BETA))
    a1 = pyro.sample('a1', dist.Normal(0, a1_sigma))   
    # create s
    s_sigma = pyro.sample('s_sigma', dist.Gamma(ALPHA, BETA))
    with pyro.plate('s_plate', n_samp):
        with pyro.plate('k_s', k):
            s = pyro.sample('s', dist.Normal(0, s_sigma))
    s = torch.transpose(s, 0, 1)
    # create d
    d_sigma = pyro.sample('d_sigma', dist.Gamma(ALPHA, BETA))
    with pyro.plate('d_plate', n_drug):
        with pyro.plate('k_d', k):
            d = pyro.sample('d', dist.Normal(0, d_sigma))
    # mat1 = sd
    mat1 = torch.matmul(s, d) # should be: n-samp x n-drug
    assert (mat1.shape[0] == n_samp) and (mat1.shape[1] == n_drug)
    mean1 = mat1[s_idx1, d_idx1] + a1
    sigma1 = pyro.sample('sigma1', dist.Gamma(ALPHA, BETA))
    with pyro.plate('data1_plate', n_obs1):
        data1 = pyro.sample('data1', dist.Normal(mean1, sigma1 * torch.ones(n_obs1)), obs=obs1)
    # create global offset
    a2_sigma = pyro.sample('a2_sigma', dist.Gamma(ALPHA, BETA))
    a2 = pyro.sample('a2', dist.Normal(0, a2_sigma))  
    # create W
    w_sigma = pyro.sample('w_sigma', dist.Gamma(ALPHA, BETA))
    with pyro.plate('w_row_plate', k):
        with pyro.plate('r_row', r):
            w_row = pyro.sample('w_row', dist.Normal(0, w_sigma))
    with pyro.plate('w_col_plate', k):
        with pyro.plate('r_col', r):
            w_col = pyro.sample('w_col', dist.Normal(0, w_sigma))
    w_col = torch.transpose(w_col, 0, 1)
    W = torch.matmul(w_col, w_row)
    assert (W.shape[0] == k) and (W.shape[1] == k)
    # Compute s'^T = Ws^T
    spr_transpose = torch.matmul(W, torch.transpose(s, 0, 1))
    spr = torch.transpose(spr_transpose, 0, 1)
    assert (spr.shape[0] == n_samp) and (spr.shape[1] == k)
    mat2 = torch.matmul(spr, d)
    assert (mat2.shape[0] == n_samp) and (mat2.shape[1] == n_drug)
    mean2 = mat2[s_idx2, d_idx2] + a2
    sigma2 = pyro.sample('sigma2', dist.Gamma(ALPHA, BETA))
    with pyro.plate('data2_plate', n_obs2):
        data2 = pyro.sample('data2', dist.Normal(mean2, sigma2 * torch.ones(n_obs2)), obs=obs2)

# Target-only model.
# n_samp: number of samples total
# n_drug: number of drugs total
# s_idx: list of sample target training indices 
# d_idx: list of drug target training indices 
# obs: list of target training observations
# n_obs: length of obs2
# k: dimension of latent vectors and W matrix 
# Note: in the paper a sample has latent vectors c and c'; here they are called s and s'.
def target_only_model(n_samp, n_drug, s_idx, d_idx, obs=None, n_obs=None, k=1):
    if obs is None and n_obs is None:
        print('Error!: both obs and n_obs are None.')
    if obs is not None:
        n_obs = obs.shape[0]
    # create global offset
    a_sigma = pyro.sample('a_sigma', dist.Gamma(ALPHA, BETA))
    a = pyro.sample('a', dist.Normal(0, a_sigma))
    # create s
    s_sigma = pyro.sample('s_sigma', dist.Gamma(ALPHA, BETA))
    with pyro.plate('s_plate', n_samp):
        with pyro.plate('k_s', k):
            s = pyro.sample('s', dist.Normal(0, s_sigma))
    # create d
    d_sigma = pyro.sample('d_sigma', dist.Gamma(ALPHA, BETA))
    with pyro.plate('d_plate', n_drug):
        with pyro.plate('k_d', k):
            d = pyro.sample('d', dist.Normal(0, d_sigma))
    # multiply s and d to create matrix
    s = torch.transpose(s, 0, 1)
    mat = torch.matmul(s, d) # should be: n-samp x n-drug
    assert (mat.shape[0] == n_samp) and (mat.shape[1] == n_drug)
    mean = mat[s_idx, d_idx] + a
    sigma = pyro.sample('sigma', dist.Gamma(ALPHA, BETA))
    with pyro.plate('data_plate', n_obs):
        data = pyro.sample('data', dist.Normal(mean, sigma * torch.ones(n_obs)), obs=obs)
    return data