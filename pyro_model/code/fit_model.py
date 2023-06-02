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

# Returns list of normally-distributed variables with means from dictionary d and constant variance "variance"
def normal_variables_from_dict(d, variance):
    var_dict = {}
    for key in d.keys():
        var_dict[key] = pyro.sample(key, dist.Normal(d[key], variance))
    return var_dict

def model(sample_list, drug_list, obs_list, sample_means, drug_means):
    num_observations = len(obs_list)
    assert len(sample_list) == num_observations
    assert len(drug_list) == num_observations
    # create variables for each sample and drug
    samples = normal_variables_from_dict(sample_means, 1)
    drugs = normal_variables_from_dict(drug_means, 1)
    # create variable for each (sample, drug) pair observed
    sigma = pyro.param("sigma", lambda: torch.ones(()), constraint=constraints.positive)
    for i in pyro.plate("data", num_observations):
        name = sample_list[i] + '_' + drug_list[i]
        mean = samples[sample_list[i]] * drugs[drug_list[i]]
        pyro.sample(name, dist.Normal(mean, sigma), obs=obs_list[i])

# given dataframe with columns 'sample', 'drug', and 'log(V_V0)_obs', return lists to pass to model
def format_for_model(d, vol_name):
    sample_list = list(d['sample'])
    drug_list = list(d['drug'])
    obs_list = []
    for obs in d[vol_name]:
        obs_list.append(torch.Tensor(obs))
    return sample_list, drug_list, obs_list

def get_means(var_list):
    means = {}
    for v in var_list:
        means[v] = 0
    return means

NUM_ARGS = 3
n = len(sys.argv)
if n != NUM_ARGS:
    print("Error: " + str(NUM_ARGS) + " arguments needed; only " + str(n) + " arguments given.")
read_fn = sys.argv[1].split("=")[1]
write_dir = sys.argv[2].split("=")[1]


smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.8.4')

pyro.enable_validation(True)
pyro.set_rng_seed(1)
logging.basicConfig(format='%(message)s', level=logging.INFO)

# Set matplotlib settings
plt.style.use('default')

df = pd.read_pickle(read_fn)
vol_name = 'log(V_V0)_obs'
sample_means = get_means(df['sample'].unique())
drug_means = get_means(df['drug'].unique())
sample_list, drug_list, obs_list = format_for_model(df, vol_name)
pyro.render_model(model, 
	model_args=(sample_list, drug_list, obs_list, sample_means, drug_means), 
	render_distributions=True, 
	filename=write_dir + '/model_diagram.png')
pyro.clear_param_store()
kernel = pyro.infer.mcmc.NUTS(model, jit_compile=True)
mcmc = pyro.infer.MCMC(kernel, num_samples=500, warmup_steps=500)
mcmc.run(sample_list, drug_list, obs_list, sample_means, drug_means)

mcmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

with open(write_dir + '/mcmc_samples.pkl', 'wb') as handle:
    pickle.dump(mcmc_samples, handle)
