import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import pyro.util
import torch
import tqdm
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
import seaborn as sns

import global_constants as const
import model_helpers as modeling
import split_helpers

NSTEPS = 10
NBINS = 20
NSEED = 3
NRANK = 3
TRAIN_FRAC = .9

def get_unique_pairs(df, col1, col2):
    a = df[[col1, col2]].drop_duplicates()
    pairs = list(zip(a[col1], a[col2]))
    assert len(pairs) == len(set(pairs))
    return pairs

def get_train_test_indices(pairs, n_train):
    n_pairs = len(pairs)
    idx = np.random.permutation(n_pairs)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    assert set(train_idx).isdisjoint(set(test_idx))
    return train_idx, test_idx

def index_pairs(pairs, indices):
    p = np.array(pairs)
    idx = np.array(indices)
    return list(map(tuple, p[idx]))

def split_by_pairs(df, col1, col2, train_pairs, test_pairs):
    df['pair'] = list(zip(df[col1], df[col2]))
    train_df = df.loc[df['pair'].isin(train_pairs)]
    test_df = df.loc[df['pair'].isin(test_pairs)]
    assert set(train_df['pair']).isdisjoint(set(test_df['pair']))
    return train_df, test_df

def split_train_test(df, n_train):
    col1 = 'sample_id'
    col2 = 'drug_id'
    # get unique sample-drug pairs
    pairs = get_unique_pairs(df, col1, col2)
    train_idx, test_idx = get_train_test_indices(pairs, n_train)
    train_pairs = index_pairs(pairs, train_idx)
    test_pairs = index_pairs(pairs, test_idx)
    return split_by_pairs(df, col1, col2, train_pairs, test_pairs)

def get_obs_info(df, obs_name):
	s = df['sample_id'].to_numpy()
	d = df['drug_id'].to_numpy()
	obs = torch.Tensor(df[obs_name].to_numpy())
	return s, d, obs

def split_data(data_fn, split_dir):
	df = pd.read_csv(data_fn)
	n_samp = df['sample_id'].nunique()
	n_drug = df['drug_id'].nunique()
	train_df, test_df = split_train_test(df, n_train)
	s_idx, d_idx, train_obs = get_obs_info(train_df)
	s_test_idx, d_test_idx, test_obs = get_obs_info(test_df)
	return n_samp, n_drug, s_idx, d_idx, train_obs, s_test_idx, d_test_idx, test_obs

def get_real_data(directory):
	train_fn = directory + '/train.pkl'
	test_fn = directory + '/test.pkl'
	sample_fn = directory + '/sample_dict.pkl'
	drug_fn = directory + '/drug_dict.pkl'
	n_samp, n_drug, s_idx, d_idx, obs_train = modeling.get_model_inputs(train_fn, sample_fn, drug_fn)
	_, _, s_test_idx, d_test_idx, obs_test = modeling.get_model_inputs(test_fn, sample_fn, drug_fn)
	obs_test = obs_test.detach().numpy()
	return n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train, obs_test

def get_synthetic_data(directory, n_total_obs=None):
	train_fn = directory + '/train.pkl'
	test_fn = directory + '/test.pkl'
	sample_fn = directory + '/sample_dict.pkl'
	drug_fn = directory + '/drug_dict.pkl'
	n_samp, n_drug, s_idx, d_idx, _ = modeling.get_model_inputs(train_fn, sample_fn, drug_fn)
	_, _, s_test_idx, d_test_idx, _ = modeling.get_model_inputs(test_fn, sample_fn, drug_fn)
	if n_total_obs != -1:
		# generate n_total_obs number of samples, with randomly chosen s_idx, d_idx
		s_idx, d_idx, s_test_idx, d_test_idx = random_indexing(n_samp, n_drug, n_total_obs)
	obs_train, obs_test = generate_data(n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx)
	return n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train, obs_test

def generate_data(n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx):
	n_train = len(s_idx)
	n_test = len(s_test_idx)
	s_indices = np.concatenate((s_idx, s_test_idx))
	d_indices = np.concatenate((d_idx, d_test_idx))
	n_obs = len(s_indices)
	obs = modeling.model(n_samp, n_drug, s_indices, d_indices, const.PARAMS, n_obs=n_obs)
	obs_train = torch.Tensor(obs.detach().numpy()[0:n_train])
	obs_test = obs.detach().numpy()[n_train:]
	assert obs_train.shape[0] == n_train
	assert obs_test.shape[0] == n_test
	return obs_train, obs_test

def random_indexing(n_samp, n_drug, n_total_obs):
	# generate sample, drug indices
	s_indices = np.random.choice(range(n_samp), size=(n_total_obs,))
	d_indices = np.random.choice(range(n_drug), size=(n_total_obs,))
	n_train = int(np.floor(const.FRACTION_TRAIN * n_total_obs))
	s_idx = s_indices[:n_train]
	s_test_idx = s_indices[n_train:]
	d_idx = d_indices[:n_train]
	d_test_idx = d_indices[n_train:]
	return s_idx, d_idx, s_test_idx, d_test_idx

def plot_histogram(read_fn, write_fn, k, use_real_data):
	fig = plt.figure()
	arr = np.loadtxt(read_fn)
	sns.histplot(arr, bins=NBINS)
	title = ', Rank: ' + str(k) + ', N: ' + str(len(arr))
	if use_real_data:
		title = 'Welm Data' + title
	else:
		title = 'Synthetic Data' + title
	plt.title(title)
	plt.xlabel('r-squared')
	plt.savefig(write_fn, bbox_inches='tight')
	plt.clf()
	plt.close()

def get_avg(read_dir, rank_list, is_test):
	if is_test:
		suffix = '_test.txt'
	else:
		suffix = '_train.txt'
	avg = []
	for k in rank_list:
		fn = read_dir + '/r_squared_' + str(k) + suffix
		arr = np.loadtxt(fn)
		avg.append(np.mean(arr))
	return avg

def plot_avg(read_dir, save_dir, use_real_data, rank_list, N):
	# get average for train and test values
	train_avg = get_avg(read_dir, rank_list, False)
	test_avg = get_avg(read_dir, rank_list, True)
	plt.plot(rank_list, train_avg, 'ko-', label='train')
	plt.plot(rank_list, test_avg, 'co-', label='test')
	title = 'Average r-squared vs rank, N = ' + str(N) + ' per point'
	if use_real_data:
		plot_fn = save_dir + '/plot_avg_real.png'
		title = 'Welm Data: ' + title
	else:
		plot_fn = save_dir + '/plot_avg_synth.png'
		title = 'Synthetic Data: ' + title
	plt.title(title)
	plt.xlabel('rank')
	plt.ylabel('average r-squared')
	plt.legend()
	write_fn = save_dir + '/avg.png'
	plt.savefig(plot_fn, bbox_inches='tight')
	plt.clf()
	plt.close()


def predict(s, d):
	mat = np.matmul(np.transpose(s), d)
	return mat

def r_squared(means, test):
    pearson_corr = np.corrcoef(test, means)
    r = pearson_corr[0, 1]
    return np.power(r, 2)

def ranks(rank_list, seed_list, use_real_data, data_dir, save_dir, obs_name):
	for k in rank_list:
		round_k(seed_list, k, use_real_data, data_dir, save_dir, obs_name)

def round_k(seed_list, k, use_real_data, data_dir, save_dir, obs_name):
	rsq_test_list = []
	rsq_train_list = []
	for seed in seed_list:
		rsq_test, rsq_train = fit_k(seed, k, use_real_data, data_dir, obs_name)
		rsq_test_list.append(rsq_test)
		rsq_train_list.append(rsq_train)
	test_fn = save_dir + '/r_squared_' + str(k) + '_test.txt'
	train_fn = save_dir + '/r_squared_' + str(k) + '_train.txt'
	np.savetxt(test_fn, np.array(rsq_test_list))
	np.savetxt(train_fn, np.array(rsq_train_list))

def get_r_squared(s_loc, d_loc, s_idx, d_idx, obs):
	mat = predict(s_loc, d_loc)
	means = mat[s_idx, d_idx]
	rsq = r_squared(means, obs)
	return rsq

def fit_k(seed, k, use_real_data, data_dir, obs_name):
    pyro.util.set_rng_seed(seed)
    pyro.clear_param_store()
    # get data
    # DO: edit data_fn and split data functions
    data_fn = data_dir + '/rep-gdsc-ctd2-clean.csv'

    df = pd.read_csv(data_fn)
    n = len(df)
    n_train = int(np.floor(n * TRAIN_FRAC))
    n_samp = df['sample_id'].nunique()
    n_drug = df['drug_id'].nunique()
    # split data
    train_df, test_df = split_train_test(df, n_train)
    s_idx, d_idx, obs_train = get_obs_info(train_df, obs_name)
    s_test_idx, d_test_idx, obs_test = get_obs_info(test_df, obs_name)
    # fit model
    adam_params = {"lr": 0.05}
    optimizer = Adam(adam_params)
    autoguide = AutoNormal(modeling.vectorized_model)
    svi = SVI(modeling.vectorized_model, autoguide, optimizer, loss=Trace_ELBO())
    for step in tqdm.trange(NSTEPS):
        svi.step(n_samp, n_drug, s_idx, d_idx, const.PARAMS, obs=obs_train, k=k)
    # retrieve values out for s and d vectors
    s_loc = pyro.param("AutoNormal.locs.s").detach().numpy()
    s_scale = pyro.param("AutoNormal.scales.s").detach().numpy()
    d_loc = pyro.param("AutoNormal.locs.d").detach().numpy()
    d_scale = pyro.param("AutoNormal.scales.d").detach().numpy()
    # get r-squared value wrt test set
    rsq_test = get_r_squared(s_loc, d_loc, s_test_idx, d_test_idx, obs_test)
    rsq_train = get_r_squared(s_loc, d_loc, s_idx, d_idx, obs_train)
    return rsq_test, rsq_train

def main():
	# TODO: move hard-coded values to inputs & create all necessary directories
    data_dir = '~/Documents/research/tansey/msk_intern/pyro_model/data'
    use_real_data = 1
    rank_list = range(1, NRANK)
    seed_list = range(0, NSEED)
    save_dir = 'results/2023-07-12/multi_eval'
    obs_name = 'REP_auc_overlap'
    ranks(rank_list, seed_list, use_real_data, data_dir, save_dir, obs_name)
    # plot histograms - break this out into a separate function
    for k in rank_list:
    	train_fn = save_dir + '/r_squared_' + str(k) + '_train.txt'
    	write_train_fn = save_dir + '/hist_' + str(k) + '_train.png'
    	plot_histogram(train_fn, write_train_fn, k, use_real_data)
    	test_fn = save_dir + '/r_squared_' + str(k) + '_test.txt'
    	write_test_fn = save_dir + '/hist_' + str(k) + '_test.png'
    	plot_histogram(test_fn, write_test_fn, k, use_real_data)
    plot_avg(save_dir, save_dir, use_real_data, rank_list, len(seed_list))
    # DO: deal with error?? In post processing??

if __name__ == "__main__":
    main()