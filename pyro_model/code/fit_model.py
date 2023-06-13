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

from fit_model_helpers import read_pickle, get_model_inputs, model

NUM_ARGS = 5

n = len(sys.argv)
if n != NUM_ARGS:
    print("Error: " + str(NUM_ARGS) + " arguments needed; only " + str(n) + " arguments given.")
train_fn = sys.argv[1].split("=")[1]
sample_fn = sys.argv[2].split("=")[1]
drug_fn = sys.argv[3].split("=")[1]
write_dir = sys.argv[4].split("=")[1]

smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.8.4')
pyro.enable_validation(True)
logging.basicConfig(format='%(message)s', level=logging.INFO)

# Set matplotlib settings
plt.style.use('default')

n_samp, n_drug, s_idx, d_idx, obs = get_model_inputs(train_fn, sample_fn, drug_fn)
n_obs = obs.shape[0]
#pyro.render_model(model, model_args=(n_samp, n_drug, s_idx, d_idx, obs, n_obs), render_params=True, 
#                  render_distributions=True, filename=write_dir + '/model_diagram.png')
pyro.clear_param_store()
kernel = pyro.infer.mcmc.NUTS(model, jit_compile=True)
mcmc = pyro.infer.MCMC(kernel, num_samples=500, warmup_steps=500)
mcmc.run(n_samp, n_drug, s_idx, d_idx, obs=obs, n_obs=n_obs)
mcmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
# write mcmc samples to file
with open(write_dir + '/mcmc_samples.pkl', 'wb') as handle:
    pickle.dump(mcmc_samples, handle)
