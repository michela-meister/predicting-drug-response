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
import split_helpers

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

def dataset_indexing(train_fn, test_fn, sample_fn, drug_fn):
	train_fn, test_fn, sample_fn, drug_fn = files_from_args(args)
	n_samp, n_drug, s_idx, d_idx, _ = modeling.get_model_inputs(train_fn, sample_fn, drug_fn)
	_, _, s_test_idx, d_test_idx, _ = modeling.get_model_inputs(test_fn, sample_fn, drug_fn)
	return s_idx, d_idx, s_test_idx, d_test_idx

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

def draw_mcmc_samples(kernel, n_samp, n_drug, s_idx, d_idx, params, obs_train, n_mcmc, n_warmup, initial_params=None, k=1):
	n_train = len(s_idx)
	mcmc = pyro.infer.MCMC(kernel, num_samples=n_mcmc, warmup_steps=n_warmup, initial_params=initial_params)
	mcmc.run(n_samp, n_drug, s_idx, d_idx, params, obs=obs_train, n_obs=n_train, k=k)
	mcmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
	return mcmc_samples

def get_mcmc_samples(n_samp, n_drug, s_idx, d_idx, params, obs_train, n_mcmc, n_warmup, initial_params=None):
	kernel = pyro.infer.mcmc.NUTS(modeling.model, jit_compile=True)
	mcmc_samples = draw_mcmc_samples(kernel, n_samp, n_drug, s_idx, d_idx, params, obs_train, n_mcmc, n_warmup)
	return mcmc_samples

def append_sample_to_dict(d, s):
	# if d is empty, initialize to s
	if len(d.keys()) == 0:
		return s
	# otherwise, d is a dictionary of numpy arrays
	# for each key, get the array, and append to the array
	for key in d.keys():
		d[key] = np.append(d[key], np.array(s[key]))
	return d

# return final sample as dictionary
# n: number of mcmc_samples
def get_final_sample(mcmc_samples, n):
	keys = mcmc_samples.keys()
	final_sample = {}
	for k in keys:
		final_sample[k] = np.array(mcmc_samples[k][n-1])
	return final_sample

def mcmc_sample_to_tensor(s):
	d = {}
	for key in s.keys():
		d[key] = torch.Tensor(s[key])
	return d

def get_thinning_indices(n_draw, n_mcmc, thinning):
	assert n_draw == n_mcmc * thinning
	idx = np.linspace(0, n_draw, n_mcmc + 1)
	idx = idx - 1
	idx = idx[1:]
	assert (np.ceil(idx) == np.floor(idx)).all()
	idx = np.array(idx, dtype=int)
	return idx

def simple_thinning(mcmc_samples, n_draw, n_mcmc, thinning):
	idx = get_thinning_indices(n_draw, n_mcmc, thinning)
	thinned_samples = {}
	for k in mcmc_samples.keys():
		arr = mcmc_samples[k]
		thinned_samples[k] = arr[idx]
	return thinned_samples

def get_mcmc_samples_with_simple_thinning(n_samp, n_drug, s_idx, d_idx, params, obs_train, n_mcmc, n_warmup, thinning):
	# number of total samples to draw, which will be thinned out
	kernel = pyro.infer.mcmc.NUTS(modeling.model, jit_compile=True)
	n_draw = n_mcmc * thinning
	mcmc_samples = draw_mcmc_samples(kernel, n_samp, n_drug, s_idx, d_idx, params, obs_train, n_draw, n_warmup, initial_params=None)
	# thin samples
	thinned_samples = simple_thinning(mcmc_samples, n_draw, n_mcmc, thinning)
	return thinned_samples

def vec_get_mcmc_samples_with_simple_thinning(kernel, n_samp, n_drug, s_idx, d_idx, params, obs_train, n_mcmc, n_warmup, thinning, k):
	# number of total samples to draw, which will be thinned out
	n_draw = n_mcmc * thinning
	mcmc_samples = draw_mcmc_samples(kernel, n_samp, n_drug, s_idx, d_idx, params, obs_train, n_draw, n_warmup, initial_params=None, k=k)
	# thin samples
	thinned_samples = simple_thinning(mcmc_samples, n_draw, n_mcmc, thinning)
	return thinned_samples

# n_mcmc is the number of samples desired
# thinning is the number of draws from the mcmc between returned samples
def get_mcmc_samples_with_thinning(n_samp, n_drug, s_idx, d_idx, params, obs_train, n_mcmc, n_warmup, thinning):
	prev_sample = None
	thinned_samples = {}
	# initialize model
	kernel = pyro.infer.mcmc.NUTS(modeling.model, jit_compile=True)
	for r in range(0, n_mcmc):
		round_warmup = 0
		# if first round, warmup
		if r == 0:
			round_warmup = n_warmup
		# get thinning number of samples normally
		mcmc_samples = draw_mcmc_samples(kernel, n_samp, n_drug, s_idx, d_idx, params, obs_train, n_mcmc=thinning, n_warmup=round_warmup, 
			initial_params=prev_sample)
		print('thinning mcmc_samples - a: ')
		print(mcmc_samples['a'])
		final_sample = get_final_sample(mcmc_samples, thinning)
		print('final sample - a: ')
		print(final_sample['a'])
		thinned_samples = append_sample_to_dict(thinned_samples, final_sample)
		prev_sample = mcmc_sample_to_tensor(final_sample)
	return thinned_samples

def evaluation(mcmc_samples, s_test_idx, d_test_idx, obs_test, hi, lo):
	mu, sigma = evaluate.predict(mcmc_samples, s_test_idx, d_test_idx)
	r_sq = evaluate.r_squared(mu, obs_test)
	coverage = evaluate.coverage(mu, sigma, obs_test, hi, lo)
	return r_sq, coverage

def vec_evaluation(mcmc_samples, s_test_idx, d_test_idx, obs_test, hi, lo, n_mcmc, n_samp, n_drug, k):
	mu, sigma = evaluate.vectorized_predict(mcmc_samples, s_test_idx, d_test_idx, n_mcmc, n_samp, n_drug, k)
	r_sq = evaluate.r_squared(mu, obs_test)
	coverage = evaluate.coverage(mu, sigma, obs_test, hi, lo)
	return r_sq, coverage

# extracts and saves args
def retrieve_args(args):
	data_dir = args[1].split("=")[1]
	n_total_obs = int(args[2].split("=")[1])
	n_mcmc = int(args[3].split("=")[1])
	n_warmup = int(args[4].split("=")[1])
	n_iter = int(args[5].split("=")[1])
	thinning = int(args[6].split("=")[1])
	directory = args[7].split("=")[1]
	use_real_data = bool(int(args[8].split("=")[1]))
	k = int(args[9].split("=")[1])
	seed = int(args[10].split("=")[1])
	input_args = {'data_dir': data_dir, 'n_total_obs': n_total_obs, 'n_mcmc': n_mcmc, 'n_warmpup': n_warmup, 'n_iter': n_iter, 'thining': thinning,
	    'directory': directory, 'use_real_data': use_real_data, 'k': k, 'seed': seed}
	helpers.write_to_pickle(input_args, directory + '/input_args.pkl')
	return data_dir, n_total_obs, n_mcmc, n_warmup, n_iter, thinning, directory, use_real_data, k, seed

def orig_main():
	helpers.check_args(sys.argv, 9)
	data_dir, n_total_obs, n_mcmc, n_warmup, n_iter, thinning, directory, use_real_data = retrieve_args(sys.argv)
    # define file fns
	r_sq_fn = directory + '/r_squared.txt'
	cov_fn = directory + '/coverage.txt'
	r_sq_list = []
	cov_list = []
	# decide whether to use real or synthetic data
	for seed in range(n_iter):
		pyro.set_rng_seed(seed)
		pyro.clear_param_store()
		# split dataset
		data_fn = data_dir + '/welm_pdx_clean_mid_volume.csv'
		split_dir = data_dir + '/split'
		split_helpers.split_dataset(data_fn, split_dir)
		if use_real_data:
			n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train, obs_test = get_real_data(split_dir)
		else:
			n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train, obs_test = get_synthetic_data(split_dir, n_total_obs)
		# fit model to synthetic data
		mcmc_samples = get_mcmc_samples_with_simple_thinning(n_samp, n_drug, s_idx, d_idx, const.PARAMS, obs_train, n_mcmc, n_warmup, thinning=thinning)
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
	histogram_r_sq(r_sq_fn, r_sq_plot_fn, use_real_data)
	histogram_coverage(cov_fn, cov_plot_fn, use_real_data)

def vectorized_main():
	helpers.check_args(sys.argv, 10)
	data_dir, n_total_obs, n_mcmc, n_warmup, n_iter, thinning, directory, use_real_data, k = retrieve_args(sys.argv)
    # define file fns
	r_sq_fn = directory + '/r_squared.txt'
	cov_fn = directory + '/coverage.txt'
	r_sq_list = []
	cov_list = []
	# decide whether to use real or synthetic data
	for seed in range(n_iter):
		pyro.set_rng_seed(seed)
		pyro.clear_param_store()
		# split dataset
		data_fn = data_dir + '/welm_pdx_clean_mid_volume.csv'
		split_dir = data_dir + '/split'
		split_helpers.split_dataset(data_fn, split_dir)
		if use_real_data:
			n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train, obs_test = get_real_data(split_dir)
		else:
			n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train, obs_test = get_synthetic_data(split_dir, n_total_obs)
		# get model
		if k > 1:
			kernel = pyro.infer.mcmc.NUTS(modeling.vectorized_model, jit_compile=True)
		if k == 1:
			kernel = pyro.infer.mcmc.NUTS(modeling.model, jit_compile=True)
		mcmc_samples = vec_get_mcmc_samples_with_simple_thinning(kernel, n_samp, n_drug, s_idx, d_idx, const.PARAMS, obs_train, n_mcmc, n_warmup, thinning=thinning, k=k)
		# evaluate vs test set
		r_sq, cov = vec_evaluation(mcmc_samples, s_test_idx, d_test_idx, obs_test, const.HI, const.LO, n_mcmc, n_samp, n_drug, k)
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
	histogram_r_sq(r_sq_fn, r_sq_plot_fn, use_real_data)
	histogram_coverage(cov_fn, cov_plot_fn, use_real_data)

def main():
	helpers.check_args(sys.argv, 11)
	data_dir, n_total_obs, n_mcmc, n_warmup, n_iter, thinning, directory, use_real_data, k, seed = retrieve_args(sys.argv)
    # define file fns
	r_sq_fn = directory + '/r_squared.txt'
	cov_fn = directory + '/coverage.txt'
	pyro.set_rng_seed(seed)
	print('SEED: ' + str(seed))
	pyro.clear_param_store()
	# split dataset
	data_fn = data_dir + '/welm_pdx_clean_mid_volume.csv'
	split_dir = data_dir + '/' + str(seed) +  '/split'
	split_helpers.split_dataset(data_fn, split_dir)
	if use_real_data:
		n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train, obs_test = get_real_data(split_dir)
	else:
		n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train, obs_test = get_synthetic_data(split_dir, n_total_obs)
	# get model
	if k > 1:
		kernel = pyro.infer.mcmc.NUTS(modeling.vectorized_model, jit_compile=True)
	elif k == 1:
		kernel = pyro.infer.mcmc.NUTS(modeling.model, jit_compile=True)
	else:
		print('Error: k <= 0!')
	mcmc_samples = vec_get_mcmc_samples_with_simple_thinning(kernel, n_samp, n_drug, s_idx, d_idx, const.PARAMS, obs_train, n_mcmc, n_warmup, thinning=thinning, k=k)
	# evaluate vs test set
	if k > 1:
		r_sq, cov = vec_evaluation(mcmc_samples, s_test_idx, d_test_idx, obs_test, const.HI, const.LO, n_mcmc, n_samp, n_drug, k)
	elif k == 1:
		r_sq, cov = evaluation(mcmc_samples, s_test_idx, d_test_idx, obs_test, const.HI, const.LO)
	else:
		print('Error: k <= 0!')
	np.savetxt(r_sq_fn, np.array([r_sq]))
	np.savetxt(cov_fn, np.array([cov]))
	print('r-squared:')
	print(r_sq)
	print('coverage:')
	print(cov)

if __name__ == "__main__":
	#orig_main()
	#vectorized_main()
	main()
