import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import sys

from scipy import stats
import global_constants as const
import eval_helpers as evaluate

NUM_ARGS = 4

n = len(sys.argv)
if n != NUM_ARGS:
    print("Error: " + str(NUM_ARGS) + " arguments needed; only " + str(n) + " arguments given.")
mcmc_samples_fn = sys.argv[1].split("=")[1]
test_fn = sys.argv[2].split("=")[1]
write_dir = sys.argv[3].split("=")[1]

# read in mcmc samples
with open(mcmc_samples_fn, 'rb') as handle:
    mcmc_samples = pickle.load(handle)
    
# read in test data
with open(test_fn, 'rb') as handle:
    test_df = pickle.load(handle)

test = test_df['log(V_V0)'].to_numpy(dtype=float)
# get mcmc_samples, s_test_idx, d_test_idx
s_test_idx = test_df['s_idx'].to_numpy()
d_test_idx = test_df['d_idx'].to_numpy()
mu, sigma = evaluate.predict(mcmc_samples, s_test_idx, d_test_idx)
r_sq = evaluate.r_squared(mu, test)
fracs = evaluate.coverage(mu, sigma, test, const.HI, const.LO)
print("fracs: " + str(fracs))
print("r_sq: " + str(r_sq))

