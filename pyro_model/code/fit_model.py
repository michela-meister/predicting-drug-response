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

NUM_ARGS = 5
A_SIGMA_INIT = 5
G_ALPHA_INIT = 10
G_BETA_INIT = 2
ALPHA_INIT = 2
BETA_INIT = 1

def read_pickle(fn):
    with open(fn, 'rb') as handle:
        obj = pickle.load(handle)
    return obj
    
def get_model_inputs(train_fn, sample_fn, drug_fn):
    df = pd.read_pickle(train_fn)
    sample_dict = read_pickle(sample_fn)
    drug_dict = read_pickle(drug_fn)
    n_samp = len(sample_dict.keys())
    n_drug = len(drug_dict.keys())
    s_idx = df['s_idx'].to_numpy()
    d_idx = df['d_idx'].to_numpy()
    obs = torch.Tensor(df['log(V_V0)'])
    return n_samp, n_drug, s_idx, d_idx, obs

# n_samp: number of samples
# n_drug: number of drugs
# obs: torch.Tensor of observations
# s_idx: numpy array where s_idx[i] is the index of the sample for the i-th observation
# d_idx: numpy array where d_idx[i] is the index of the drug for the i-th observation
def model(n_samp, n_drug, s_idx, d_idx, obs=None, n_obs=None):
    if n_obs == None:
        n_obs = obs.shape[0]
    # create global offset
    a_sigma = pyro.param('a_sigma', torch.Tensor([A_SIGMA_INIT]), constraint=constraints.positive)
    a = pyro.sample('a', dist.Normal(torch.zeros(()), a_sigma * torch.ones(())))   
    # create s
    s_g_alpha = pyro.param('s_g_alpha', torch.Tensor([G_ALPHA_INIT]), constraint=constraints.positive)
    s_g_beta = pyro.param('s_g_beta', torch.Tensor([G_BETA_INIT]), constraint=constraints.positive)
    s_sigma = pyro.param('s_sigma', dist.Gamma(s_g_alpha, s_g_beta), constraint=constraints.positive)
    a_s_sigma = pyro.param('a_s_sigma', torch.Tensor([A_SIGMA_INIT]), constraint=constraints.positive)
    with pyro.plate('s_plate', n_samp):
        a_s = pyro.sample('a_s', dist.Normal(torch.zeros(n_samp), a_s_sigma * torch.ones(n_samp)))
        s = pyro.sample('s', dist.Normal(torch.zeros(n_samp), s_sigma * torch.ones(n_samp)))
    # create d
    d_g_alpha = pyro.param('d_g_alpha', torch.Tensor([G_ALPHA_INIT]), constraint=constraints.positive)
    d_g_beta = pyro.param('d_g_beta', torch.Tensor([G_BETA_INIT]), constraint=constraints.positive)
    d_sigma = pyro.param('d_sigma', dist.Gamma(d_g_alpha, d_g_beta), constraint=constraints.positive)
    a_d_sigma = pyro.param('a_d_sigma', torch.Tensor([A_SIGMA_INIT]), constraint=constraints.positive)
    with pyro.plate('d_plate', n_drug):
        a_d = pyro.sample('a_d', dist.Normal(torch.zeros(n_drug), a_d_sigma * torch.ones(n_drug)))
        d = pyro.sample('d', dist.Normal(torch.zeros(n_drug), d_sigma))
    # create data
    mean = s[s_idx] * d[d_idx] + a_s[s_idx] + a_d[d_idx] + a
    sigma_g_alpha = pyro.param('sigma_g_alpha', torch.Tensor([ALPHA_INIT]), constraint=constraints.positive)
    sigma_g_beta = pyro.param('sigma_g_beta', torch.Tensor([BETA_INIT]), constraint=constraints.positive)
    sigma = pyro.sample('sigma', dist.Gamma(sigma_g_alpha, sigma_g_beta))
    with pyro.plate('data_plate', n_obs):
        pyro.sample('data', dist.Normal(mean, sigma * torch.ones(n_obs)), obs=obs)

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
pyro.render_model(model, model_args=(n_samp, n_drug, s_idx, d_idx, obs, n_obs), render_params=True, 
                  render_distributions=True, filename=write_dir + '/model_diagram.png')
pyro.clear_param_store()
kernel = pyro.infer.mcmc.NUTS(model, jit_compile=True)
mcmc = pyro.infer.MCMC(kernel, num_samples=500, warmup_steps=500)
mcmc.run(n_samp, n_drug, s_idx, d_idx, obs, n_obs)
mcmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
# write mcmc samples to file
with open(write_dir + '/mcmc_samples.pkl', 'wb') as handle:
    pickle.dump(mcmc_samples, handle)
