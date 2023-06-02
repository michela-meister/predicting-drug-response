import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import sys

from scipy import stats

n_rounds = 50
n_synth = 200
lo = .05
hi = .95
nbins = 20
NUM_ARGS = 4

def sample_parameters(mcmc_samples):
    # create dictionary
    params = {}
    for k in mcmc_samples.keys():
        params[k] = np.random.choice(mcmc_samples[k])
    return params

def get_sample_drug_pairs(df):
    return list(df[['sample', 'drug']].apply(tuple, axis=1).drop_duplicates())

def synthetic_samples(lo, hi, sample, drug, params, n_synth):
    mean = params[sample] * params[drug]
    synth = np.random.normal(loc=mean, scale=1.0, size=(n_synth,))
    synth_mean = np.mean(synth)
    # get lo and hi percentile samples
    sorted_synth = sorted(synth)
    low_idx = int(np.ceil(lo * n_synth))
    hi_idx = int(np.floor(hi * n_synth))
    low_bd = sorted_synth[low_idx]
    hi_bd = sorted_synth[hi_idx]
    return low_bd, hi_bd, synth_mean

def get_emp_vals(df, sample, drug):
    return list(test_df.loc[(test_df['sample'] == sample) & (test_df['drug'] == drug)]['log(V_V0)'])

def count_in_bounds(arr, low_bd, hi_bd):
    return np.sum((arr > low_bd) & (arr < hi_bd))

# squared pearson correlation
def get_r_squared(empirical, synthetic):
    mat = np.corrcoef(empirical, synthetic)
    return np.power(mat[0, 1], 2)

def sampling_round(df, pairs, mcmc_samples, n_synth, lo, hi):
    params = sample_parameters(mcmc_samples)
    # sample mcmc parameters ---> return a dictionary of all samples
    # mcmc_samples looks like a dictionary. Keys are drugs and samples. For each key, draw a value
    # return dictionary of drawn values
    empirical = []
    synthetic = []
    in_bds = []
    # for each (sample, drug) pair, generate synthetic samples and compare w empirical data 
    for (sample, drug) in pairs:
        # generate synthetic samples for given (sample, drug) pair
        lo_bd, hi_bd, synth_mean = synthetic_samples(lo, hi, sample, drug, params, n_synth)
        emp_vals = get_emp_vals(df, sample, drug)
        empirical += emp_vals
        synthetic += [synth_mean] * len(emp_vals)
        in_bds.append(count_in_bounds(emp_vals, lo_bd, hi_bd))
    assert len(empirical) == len(df)
    assert len(in_bds) == len(pairs)
    r_sq = get_r_squared(empirical, synthetic)
    frac_in_bds = np.sum(in_bds) * 1.0 / len(empirical)
    return r_sq, frac_in_bds

def run_rounds(test_df, pairs, mcmc_samples, n_rounds, n_synth):
    r_sq_list = []
    frac_list = []
    for _ in range(0, n_rounds):
        r_sq, total_in_bds = sampling_round(test_df, pairs, mcmc_samples, n_synth, lo, hi)
        r_sq_list.append(r_sq)
        frac_list.append(total_in_bds)
    return r_sq_list, frac_list

def create_text(x, name, n_rounds, lo, hi):
    mean = np.mean(x)
    std = np.std(x)
    count = len(x)
    if name == 'r_squared':
        title = 'R^2 values: N = ' + str(n_rounds)
    elif name == 'in_bounds':
        title = 'Fraction of emp. observations w/in ' + str(100 * lo) + 'th to ' + str(100 * hi) + 'th percentiles.\nN = ' + str(n_rounds)
    else:
        print('Error! Name must be r_squared or in_bounds.')
    text = 'mean: ' + str(round(mean, 3)) + ', std: ' + str(round(std, 3)) + '\n'
    return title, text

def plot_histogram(fn, x, name, nbins, n_rounds, lo, hi):
    fig = plt.figure()
    sns.histplot(x, stat='density', bins=nbins)
    title, text = create_text(x, name, n_rounds, lo, hi)
    plt.title(title)
    fig.text(.5, -.05, text, ha='center')
    plt.savefig(fn, bbox_inches='tight')
    plt.clf()
    plt.close()

n = len(sys.argv)
if n != NUM_ARGS:
    print("Error: " + str(NUM_ARGS) + " arguments needed; only " + str(n) + " arguments given.")
samples_fn = sys.argv[1].split("=")[1]
test_fn = sys.argv[2].split("=")[1]
write_dir = sys.argv[3].split("=")[1]

# read in mcmc samples
with open(samples_fn, 'rb') as handle:
    mcmc_samples = pickle.load(handle)
    
# reach in test data
with open(test_fn, 'rb') as handle:
    test_df = pickle.load(handle)

# expand test_df so each row corresponds to a *single* observation
test_df = test_df[['sample', 'drug', 'log(V_V0)_obs']]
test_df = test_df.explode('log(V_V0)_obs').reset_index(drop=True)
test_df = test_df.rename(columns = {'log(V_V0)_obs': 'log(V_V0)'})
pairs = get_sample_drug_pairs(test_df)

r_sq_list, in_bds_list = run_rounds(test_df, pairs, mcmc_samples, n_rounds, n_synth)
print(write_dir)
plot_histogram(write_dir + '/r_squared.png', r_sq_list, 'r_squared', nbins, n_rounds, lo, hi)
plot_histogram(write_dir + '/in_bounds.png', in_bds_list, 'in_bounds', nbins, n_rounds, lo, hi)