import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import sys

from scipy import stats

def predict(mcmc_samples, s_test_idx, d_test_idx):
    assert len(s_test_idx) == len(d_test_idx)
    n = len(s_test_idx)
    # read in mcmc samples for each variable
    s = np.array(mcmc_samples['s']) 
    d = np.array(mcmc_samples['d'])
    a = np.array(mcmc_samples['a'])
    a_s = np.array(mcmc_samples['a_s'])
    a_d = np.array(mcmc_samples['a_d'])
    sigma = np.array(mcmc_samples['sigma'])
    # combine above matrices to create mu
    m = s.shape[0]
    mu = np.multiply(s[0:m, s_test_idx], d[0:m, d_test_idx]) + a_s[0:m, s_test_idx] + a_d[0:m, d_test_idx] + a
    assert (mu.shape[0] == m) and (mu.shape[1] == n)
    assert (sigma.shape[0] == m) and (sigma.shape[1] == 1)
    return mu, sigma

def vectorized_predict(mcmc_samples, s_test_idx, d_test_idx, n_mcmc, n_samp, n_drug, k):
    assert len(s_test_idx) == len(d_test_idx)
    n = len(s_test_idx)
    m = n_mcmc
    # read in mcmc samples for each variable
    s = np.array(mcmc_samples['s']) 
    d = np.array(mcmc_samples['d'])
    a = np.array(mcmc_samples['a'])
    sigma = np.array(mcmc_samples['sigma'])
    # sanity check 
    print('S shape: ')
    print(s.shape)
    print('m: ' + str(n_mcmc))
    print('k: ' + str(k))
    print('n_samp: ' + str(n_samp))
    assert (s.shape[0] == m) and (s.shape[1] == k) and (s.shape[2] == n_samp)
    assert (d.shape[0] == m) and (d.shape[1] == k) and (d.shape[2] == n_drug) 
    assert (a.shape[0] == m) and (a.shape[1] == 1)
    # combine above matrices to create mu
    # to create mu, multiply s and d together
    # get array with n
    s = np.transpose(s, (0, 2, 1))
    mat = np.matmul(s, d)
    mu = mat[:, s_test_idx, d_test_idx] + a
    assert (mu.shape[0] == m) and (mu.shape[1] == n)
    assert (sigma.shape[0] == m) and (sigma.shape[1] == 1)
    return mu, sigma

def r_squared(mu, test):
    means = np.mean(mu, axis=0)
    assert means.shape[0] == test.shape[0]
    pearson_corr = np.corrcoef(test, means)
    r = pearson_corr[0, 1]
    return np.power(r, 2)

# function to compute coverage
def coverage(mu, sigma, obs, hi, lo):
    # generate synthetic samples from normal distribution with mean mu
    m = mu.shape[0]
    n = mu.shape[1]
    # generate synthetic samples for each observation
    synth = mu + sigma * np.random.normal(loc=0, scale=1, size=(m, n))
    # sort synthetic samples for each observation
    sorted_synth = np.sort(synth, axis=0)
    # compute hi and lo index
    lo_idx = int(np.ceil(lo * m))
    hi_idx = int(np.floor(hi * m))
    # get synthetic samples at hi and lo indices
    lo_bound = sorted_synth[lo_idx, :]
    hi_bound = sorted_synth[hi_idx, :]
    # is obs in [hi, lo]?
    frac = np.sum(np.logical_and(lo_bound < obs, obs < hi_bound) / (1.0 * len(obs)))
    return frac