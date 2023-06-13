import logging
import numpy as np
import os
import pickle
import pyro
import sys
import torch
from scipy import stats
import pyro.distributions as dist
import pyro.distributions.constraints as constraints

from fit_model import read_pickle, get_model_inputs, model
from evaluate_model import predict, r_squared, coverage

NUM_ARGS = 5
A_SIGMA_INIT = 5
G_ALPHA_INIT = 10
G_BETA_INIT = 2
ALPHA_INIT = 2
BETA_INIT = 1
LO = .05
HI = .95

n = len(sys.argv)
if n != NUM_ARGS:
    print("Error: " + str(NUM_ARGS) + " arguments needed; only " + str(n) + " arguments given.")
# read in model inputs
train_fn = sys.argv[1].split("=")[1]
test_fn = sys.argv[2].split("=")[1]
sample_fn = sys.argv[3].split("=")[1]
drug_fn = sys.argv[4].split("=")[1]
n_samp, n_drug, s_idx, d_idx, _ = get_model_inputs(train_fn, sample_fn, drug_fn)
_, _, s_test_idx, d_test_idx, _ = get_model_inputs(test_fn, sample_fn, drug_fn)
# generate synthetic data
n_train = len(s_idx)
n_test = len(s_test_idx)
total_obs = n_train + n_test
s_indices = np.concatenate((s_idx, s_test_idx))
d_indices = np.concatenate((d_idx, d_test_idx))
obs = model(n_samp, n_drug, s_indices, d_indices, n_obs=total_obs)
obs_train = obs[0:n_train]
obs_test = obs[n_train:]
# fit model
pyro.clear_param_store()
kernel = pyro.infer.mcmc.NUTS(model, jit_compile=True)
mcmc = pyro.infer.MCMC(kernel, num_samples=500, warmup_steps=500)
mcmc.run(n_samp, n_drug, s_idx, d_idx, obs=obs_train, n_obs=obs_train.shape[0])
mcmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
# evaluate model
test = obs_test.numpy()
mu, sigma = predict(mcmc_samples, s_test_idx, d_test_idx)
r_sq = r_squared(mu, test)
fracs = coverage(mu, sigma, test, HI, LO)
print("fracs: " + str(fracs))
print("r_sq: " + str(r_sq))