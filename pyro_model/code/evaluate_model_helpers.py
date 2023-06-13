import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import sys

from scipy import stats

NBINS = 20

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
    # TODO: Figure out how to get correct variance in here
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