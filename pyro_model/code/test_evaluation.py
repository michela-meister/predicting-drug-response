import matplotlib.pyplot as plt
import numpy as np 
import pyro
import seaborn as sns
import sys
import torch

import global_constants as const
import helpers
import model_helpers as modeling
import eval_helpers as evaluate

def files_from_args(args):
	train_fn = args[1].split("=")[1]
	test_fn = args[2].split("=")[1]
	sample_fn = args[3].split("=")[1]
	drug_fn = args[4].split("=")[1]
	return train_fn, test_fn, sample_fn, drug_fn

def params_from_args(args):
	n_total_obs = int(args[5].split("=")[1])
	if n_total_obs == -1:
		n_total_obs = None
	n_mcmc = int(args[6].split("=")[1])
	n_warmup = int(args[7].split("=")[1])
	n_iter = int(args[8].split("=")[1])
	directory = args[9].split("=")[1]
	return n_total_obs, n_mcmc, n_warmup, n_iter, directory

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

def dataset_indexing(args):
	train_fn, test_fn, sample_fn, drug_fn = files_from_args(args)
	n_samp, n_drug, s_idx, d_idx, _ = modeling.get_model_inputs(train_fn, sample_fn, drug_fn)
	_, _, s_test_idx, d_test_idx, _ = modeling.get_model_inputs(test_fn, sample_fn, drug_fn)
	return n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx

def random_indexing(args, n_total_obs):
	train_fn, test_fn, sample_fn, drug_fn = files_from_args(args)
	n_samp, n_drug, _, _, _ = modeling.get_model_inputs(train_fn, sample_fn, drug_fn)
	# generate sample, drug indices
	s_indices = np.random.choice(range(n_samp), size=(n_total_obs,))
	d_indices = np.random.choice(range(n_drug), size=(n_total_obs,))
	n_train = int(np.floor(const.FRACTION_TRAIN * n_total_obs))
	s_idx = s_indices[:n_train]
	s_test_idx = s_indices[n_train:]
	d_idx = d_indices[:n_train]
	d_test_idx = d_indices[n_train:]
	return n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx

def get_synthetic_data(args, n_total_obs=None):
	if n_total_obs is not None:
		# generate n_total_obs number of samples, with randomly chosen s_idx, d_idx
		n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx = random_indexing(args, n_total_obs)
	else:
		n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx = dataset_indexing(args)
	obs_train, obs_test = generate_data(n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx)
	return n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train, obs_test

def fit_to_model(n_samp, n_drug, s_idx, d_idx, params, obs_train, n_mcmc, n_warmup):
	n_train = len(s_idx)
	pyro.clear_param_store()
	kernel = pyro.infer.mcmc.NUTS(modeling.model, jit_compile=True)
	mcmc = pyro.infer.MCMC(kernel, num_samples=n_mcmc, warmup_steps=n_warmup)
	mcmc.run(n_samp, n_drug, s_idx, d_idx, params, obs=obs_train, n_obs=n_train)
	mcmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
	return mcmc_samples

def fit_to_model_with_thinning(n_samp, n_drug, s_idx, d_idx, params, obs_train, n_mcmc, n_warmup, n_thin):
	# n_samp is the number of samples desired
	# n_thin is the thinning
	prev_sample = None
	d = {}
	for i in range(n_samp):
		mcmc.run(n_samp, n_drug, s_idx, d_idx, params, obs=obs_train, n_obs=n_train, initial_params=prev_sample)
		mcmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
		sample = get_final_sample(mcmc_samples)
		append_sample(d, sample)
		prev_sample = sample
	# read in n_thin samples
	# save the last sample
	# initialize the next run with this saved sample

def evaluation(mcmc_samples, s_test_idx, d_test_idx, obs_test, hi, lo):
	mu, sigma = evaluate.predict(mcmc_samples, s_test_idx, d_test_idx)
	r_sq = evaluate.r_squared(mu, obs_test)
	coverage = evaluate.coverage(mu, sigma, obs_test, hi, lo)
	return r_sq, coverage

def histogram_r_sq(r_sq_fn, write_fn):
	r_sq_list = np.loadtxt(r_sq_fn)
	fig = plt.figure()
	sns.histplot(r_sq_list)
	plt.title('r-squared values')
	plt.savefig(write_fn, bbox_inches='tight')
	plt.clf()
	plt.close()

def histogram_coverage(cov_fn, write_fn):
	cov_list = np.loadtxt(cov_fn)
	fig = plt.figure()
	sns.histplot(cov_list)
	plt.title('coverage')
	plt.savefig(write_fn, bbox_inches='tight')
	plt.clf()
	plt.close()

def save_args(args):
	train_fn, test_fn, sample_fn, drug_fn = files_from_args(args)
	n_total_obs, n_mcmc, n_warmup, n_iter, directory = params_from_args(sys.argv)
	input_args = {'train_fn': train_fn, 'test_fn': test_fn, 'sample_fn': sample_fn, 'drug_fn': drug_fn, 
	    'n_total_obs': n_total_obs, 'n_mcmc': n_mcmc, 'n_warmup': n_warmup, 'n_iter': n_iter, 'directory': directory}
	helpers.write_to_pickle(input_args, directory + '/input_args.pkl')

def main():
	helpers.check_args(sys.argv, 10)
	save_args(sys.argv)
	n_total_obs, n_mcmc, n_warmup, n_iter, directory = params_from_args(sys.argv)
	r_sq_fn = directory + '/r_squared.txt'
	cov_fn = directory + '/coverage.txt'
	r_sq_list = []
	cov_list = []
	for seed in range(n_iter):
		pyro.set_rng_seed(seed)
		# generate synthetic data
		n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train, obs_test = get_synthetic_data(sys.argv, n_total_obs)
		# fit model to synthetic data
		mcmc_samples = fit_to_model(n_samp, n_drug, s_idx, d_idx, const.PARAMS, obs_train, n_mcmc, n_warmup)
		# evaluate vs test set
		r_sq, cov = evaluation(mcmc_samples, s_test_idx, d_test_idx, obs_test, const.HI, const.LO)
		r_sq_list.append(r_sq)
		cov_list.append(cov)
		np.savetxt(r_sq_fn, np.array(r_sq_list))
		np.savetxt(cov_fn, np.array(cov_list))
	print('r-squared:')
	print(r_sq_list)
	print('coverage:')
	print(cov_list)
	r_sq_plot_fn = directory + '/r_squared_plot.png'
	cov_plot_fn = directory + '/coverage_plot.png'
	histogram_r_sq(r_sq_fn, r_sq_plot_fn)
	histogram_coverage(cov_fn, cov_plot_fn)

if __name__ == "__main__":
    main()