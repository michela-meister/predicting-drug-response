import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import pyro
import pyro.distributions as dist
import pyro.util
import torch
import tqdm
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
import seaborn as sns
import sys

import global_constants as const
import model_helpers as modeling
import split_helpers

NSTEPS = 1000
NBINS = 20
NSEED = 1
NRANK = 1
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

def split_from_fold(df, fold_fn, model_seed):
	# read in split (list of arrays) from fold
	# index into arrays with model_seed to get fold
	# return train, test based on fold
	return train_df, test_df

def split_train_test(df):
    col1 = 'sample_id'
    col2 = 'drug_id'
    # get unique sample-drug pairs
    pairs = get_unique_pairs(df, col1, col2)
    n_train = int(np.ceil(len(pairs) * TRAIN_FRAC))
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
	train_df, test_df = split_train_test(df)
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

def plot_histogram(read_fn, write_fn, k, use_real_data, dataset):
	fig = plt.figure()
	arr = np.loadtxt(read_fn)
	sns.histplot(arr, bins=NBINS)
	title = ', K: ' + str(k) + ', N: ' + str(len(arr))
	if use_real_data:
		title = dataset + title
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

def pearson_corr(means, test):
	pearson_corr = np.corrcoef(test, means)
	return pearson_corr[0, 1]

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

def predict_mat2(s, d, w_row, w_col, k, r):
	W = np.matmul(np.transpose(w_col), w_row)
	# s already comes transposed, as defined in model
	assert s.shape[0] == k
	assert W.shape[0] == k and W.shape[0] == k
	s_prime = np.matmul(W, s) 
	mat2 = np.matmul(np.transpose(s_prime), d)
	return mat2

def transfer_fit_k(split_seed, model_seed, data_dir, k, r, obs_name1, obs_name2, n_steps):
    pyro.util.set_rng_seed(split_seed)
    pyro.clear_param_store()
    # get data
    # DO: edit data_fn and split data functions
    data_fn = data_dir + '/rep-gdsc-ctd2-mean-log.csv'
    df = pd.read_csv(data_fn)
    n_samp = df['sample_id'].nunique()
    n_drug = df['drug_id'].nunique()
    # get base layer info
    s_idx1, d_idx1, obs_1 = get_obs_info(df, obs_name1)
    # split data
    train_df, test_df = split_train_test(df)
    s_idx2, d_idx2, obs_2 = get_obs_info(train_df, obs_name2)
    s_test_idx, d_test_idx, obs_test = get_obs_info(test_df, obs_name2)
    # fit model
    pyro.util.set_rng_seed(model_seed)
    adam_params = {"lr": 0.05}
    optimizer = Adam(adam_params)
    autoguide = AutoNormal(modeling.transfer_model)
    svi = SVI(modeling.transfer_model, autoguide, optimizer, loss=Trace_ELBO())
    losses = []
    for step in tqdm.trange(n_steps):
        svi.step(n_samp, n_drug, s_idx1, d_idx1, s_idx2, d_idx2, obs_1, len(obs_1), obs_2, len(obs_2), r=r, k=k)
        loss = svi.evaluate_loss(n_samp, n_drug, s_idx1, d_idx1, s_idx2, d_idx2, obs_1, len(obs_1), obs_2, len(obs_2), r=r, k=k)
        losses.append(loss)
    print('FINAL LOSS DIFF: ' + str(losses[len(losses) - 1] - losses[len(losses) - 2]))
    # retrieve values out for s and d vectors
    s_loc = pyro.param("AutoNormal.locs.s").detach().numpy()
    s_scale = pyro.param("AutoNormal.scales.s").detach().numpy()
    d_loc = pyro.param("AutoNormal.locs.d").detach().numpy()
    d_scale = pyro.param("AutoNormal.scales.d").detach().numpy()
    # need to retrive w_col, w_row and reconstruct s'!
    w_row_loc = pyro.param("AutoNormal.locs.w_row").detach().numpy()
    w_col_loc = pyro.param("AutoNormal.locs.w_col").detach().numpy()
    # predict function: takes in w_row, w_col, s, d --> mat2
    mat2 = predict_mat2(s_loc, d_loc, w_row_loc, w_col_loc, k, r)
    # eval test rsq
    test_means = mat2[s_test_idx, d_test_idx]
    corr_test = pearson_corr(test_means, obs_test.numpy())
    # eval train rsq
    train_means = mat2[s_idx2, d_idx2]
    corr_train = pearson_corr(train_means, obs_2.numpy())
    return corr_test, corr_train

def get_args(args, n):
	if len(args) != n + 1:
		print('Expected ' + str(n + 1) + ' arguments, but got ' + str(len(args)))
	split_seed = int(args[1].split("=")[1])
	model_seed = int(args[2].split("=")[1])
	k = int(args[3].split("=")[1])
	r = int(args[4].split("=")[1])
	obs_name1 = args[5].split("=")[1]
	obs_name2 = args[6].split("=")[1]
	save_dir = args[7].split("=")[1]
	n_steps = int(args[8].split("=")[1])
	data_dir = args[9].split("=")[1]
	return split_seed, model_seed, k, r, obs_name1, obs_name2, save_dir, n_steps, data_dir

def main():
	split_seed, model_seed, k, r, obs_name1, obs_name2, save_dir, n_steps, data_dir = get_args(sys.argv, 9)
	#data_dir = '~/Documents/research/tansey/msk_intern/pyro_model/data'
	corr_test, corr_train = transfer_fit_k(split_seed, model_seed, data_dir, k, r, obs_name1, obs_name2, n_steps)
	save_fn = save_dir + '/' + str(model_seed) + '.pkl'
	print('corr_test: ' + str(corr_test))
	print('corr_train: ' + str(corr_train))
	vals_dict = {'split_seed': split_seed, 'model_seed': model_seed, 'k': k, 'r': r, 'obs_name1': obs_name1, 'obs_name2': obs_name2, 'n_steps': n_steps, 'corr_train': corr_train, 'corr_test': corr_test}
	with open(save_fn, 'wb') as handle:
		pickle.dump(vals_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# for a set seed, r, k, fn: fit the model
# compute rsq_test, rsq_train: between predict and original
# save to file

if __name__ == "__main__":
    main()