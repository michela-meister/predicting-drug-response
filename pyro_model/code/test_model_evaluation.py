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

import global_constants as const
import model_helpers as modeling
import eval_helpers as evaluate

NUM_ARGS = 5

n = len(sys.argv)
if n != NUM_ARGS:
    print("Error: " + str(NUM_ARGS) + " arguments needed; only " + str(n) + " arguments given.")
# read in model inputs
train_fn = sys.argv[1].split("=")[1]
test_fn = sys.argv[2].split("=")[1]
sample_fn = sys.argv[3].split("=")[1]
drug_fn = sys.argv[4].split("=")[1]
n_samp, n_drug, s_idx, d_idx, _ = modeling.get_model_inputs(train_fn, sample_fn, drug_fn)
_, _, s_test_idx, d_test_idx, _ = modeling.get_model_inputs(test_fn, sample_fn, drug_fn)
# generate synthetic data
n_train = len(s_idx)
n_test = len(s_test_idx)
total_obs = n_train + n_test
s_indices = np.concatenate((s_idx, s_test_idx))
d_indices = np.concatenate((d_idx, d_test_idx))
obs = modeling.model(n_samp, n_drug, s_indices, d_indices, const.PARAMS, n_obs=total_obs)
obs_train = torch.Tensor(obs.detach().numpy()[0:n_train])
obs_test = obs.detach().numpy()[n_train:]
#fit model
pyro.clear_param_store()
kernel = pyro.infer.mcmc.NUTS(modeling.model, jit_compile=True)
mcmc = pyro.infer.MCMC(kernel, num_samples=50000, warmup_steps=500)
mcmc.run(n_samp, n_drug, s_idx, d_idx, const.PARAMS, obs=obs_train, n_obs=n_train)
mcmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
# evaluate model
test = obs_test
mu, sigma = evaluate.predict(mcmc_samples, s_test_idx, d_test_idx)
r_sq = evaluate.r_squared(mu, test)
fracs = evaluate.coverage(mu, sigma, test, const.HI, const.LO)
print("fracs: " + str(fracs))
print("r_sq: " + str(r_sq))