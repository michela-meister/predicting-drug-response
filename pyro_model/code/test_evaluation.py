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

def get_real_data(directory):
	train_fn = directory + '/train.pkl'
	test_fn = directory + '/test.pkl'
	sample_fn = directory + '/sample_dict.pkl'
	drug_fn = directory + '/drug_dict.pkl'
	n_samp, n_drug, s_idx, d_idx, obs_train = modeling.get_model_inputs(train_fn, sample_fn, drug_fn)
	_, _, s_test_idx, d_test_idx, obs_test = modeling.get_model_inputs(test_fn, sample_fn, drug_fn)
	obs_test = obs_test.detach().numpy()
	return n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train, obs_test

def get_synthetic_data(args, n_total_obs=None):
	if n_total_obs is not None:
		# generate n_total_obs number of samples, with randomly chosen s_idx, d_idx
		n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx = random_indexing(args, n_total_obs)
	else:
		n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx = dataset_indexing(args)
	obs_train, obs_test = generate_data(n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx)
	return n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train, obs_test

def draw_mcmc_samples(kernel, n_samp, n_drug, s_idx, d_idx, params, obs_train, n_mcmc, n_warmup, initial_params=None):
	n_train = len(s_idx)
	mcmc = pyro.infer.MCMC(kernel, num_samples=n_mcmc, warmup_steps=n_warmup, initial_params=initial_params)
	mcmc.run(n_samp, n_drug, s_idx, d_idx, params, obs=obs_train, n_obs=n_train)
	mcmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
	return mcmc_samples

def get_mcmc_samples(n_samp, n_drug, s_idx, d_idx, params, obs_train, n_mcmc, n_warmup, initial_params=None):
	kernel = pyro.infer.mcmc.NUTS(modeling.model, jit_compile=True)
	mcmc_samples = draw_mcmc_samples(kernel, n_samp, n_drug, s_idx, d_idx, params, obs_train, n_mcmc, n_warmup)
	return mcmc_samples

# def mcmc_samples(n_samp, n_drug, s_idx, d_idx, params, obs_train, n_mcmc, n_warmup):
# 	n_train = len(s_idx)
# 	pyro.clear_param_store()
# 	kernel = pyro.infer.mcmc.NUTS(modeling.model, jit_compile=True)
# 	mcmc = pyro.infer.MCMC(kernel, num_samples=n_mcmc, warmup_steps=n_warmup)
# 	mcmc.run(n_samp, n_drug, s_idx, d_idx, params, obs=obs_train, n_obs=n_train)
# 	mcmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
# 	return mcmc_samples

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

def main_synth_data():
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
		mcmc_samples = get_mcmc_samples_with_simple_thinning(n_samp, n_drug, s_idx, d_idx, const.PARAMS, obs_train, n_mcmc, n_warmup, thinning=2)
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
	r_sq_plot_fn = directory + '/synth_r_squared_plot.png'
	cov_plot_fn = directory + '/synth_coverage_plot.png'
	histogram_r_sq(r_sq_fn, r_sq_plot_fn)
	histogram_coverage(cov_fn, cov_plot_fn)

def main_real_data():
	args = sys.argv
	helpers.check_args(args, 8)
	#save_args(sys.argv)
	# read in args
	data_dir = args[1].split("=")[1]
	n_total_obs = int(args[2].split("=")[1])
	n_mcmc = int(args[3].split("=")[1])
	n_warmup = int(args[4].split("=")[1])
	n_iter = int(args[5].split("=")[1])
	thinning = int(args[6].split("=")[1])
	directory = args[7].split("=")[1]
    # define file fns
	r_sq_fn = directory + '/r_squared.txt'
	cov_fn = directory + '/coverage.txt'
	r_sq_list = []
	cov_list = []
	for seed in range(n_iter):
		pyro.set_rng_seed(seed)
		pyro.clear_param_store()
		# split dataset
		data_fn = data_dir + '/welm_pdx_clean_mid_volume.csv'
		split_dir = data_dir + '/split'
		split_helpers.split_dataset(data_fn, split_dir)
		# get real data
		n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train, obs_test = get_real_data(split_dir)
		# fit model to synthetic data
		mcmc_samples = get_mcmc_samples_with_simple_thinning(n_samp, n_drug, s_idx, d_idx, const.PARAMS, obs_train, n_mcmc, n_warmup, thinning=2)
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

def simply_get_mcmc_samples():
	helpers.check_args(sys.argv, 10)
	save_args(sys.argv)
	n_total_obs, n_mcmc, n_warmup, n_iter, directory = params_from_args(sys.argv)
	# get unthinned samples of length 10
	pyro.set_rng_seed(0)
	pyro.clear_param_store()
	print('random val: ' + str(np.random.randn()))
	n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train, obs_test = get_synthetic_data(sys.argv, n_total_obs)
	mcmc_thinned1 = get_mcmc_samples(n_samp, n_drug, s_idx, d_idx, const.PARAMS, obs_train, 5, n_warmup=10, initial_params=None)
	return mcmc_thinned1

def test_mcmc_ordering():
	helpers.check_args(sys.argv, 10)
	save_args(sys.argv)
	n_total_obs, n_mcmc, n_warmup, n_iter, directory = params_from_args(sys.argv)
	# get unthinned samples of length 10
	pyro.set_rng_seed(0)
	pyro.clear_param_store()
	print('random val: ' + str(np.random.randn()))
	n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train, obs_test = get_synthetic_data(sys.argv, n_total_obs)
	mcmc_thinned1 = get_mcmc_samples(n_samp, n_drug, s_idx, d_idx, const.PARAMS, obs_train, 5, n_warmup=10, initial_params=None)
	# get un-thinned samples
	pyro.set_rng_seed(0)
	pyro.clear_param_store()
	print('random val: ' + str(np.random.randn()))
	n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train, obs_test = get_synthetic_data(sys.argv, n_total_obs)
	mcmc_thinned2 = get_mcmc_samples(n_samp, n_drug, s_idx, d_idx, const.PARAMS, obs_train, 10, n_warmup=10, initial_params=None)	
	print('mcmc_thinned1: ')
	print(mcmc_thinned1)
	print('mcmc_thinned2: ')
	print(mcmc_thinned2)

def test_mcmc_state():
	helpers.check_args(sys.argv, 10)
	save_args(sys.argv)
	n_total_obs, n_mcmc, n_warmup, n_iter, directory = params_from_args(sys.argv)
	# get two sets of 5 samples
	pyro.set_rng_seed(0)
	pyro.clear_param_store()
	print('random val: ' + str(np.random.randn()))
	n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train, obs_test = get_synthetic_data(sys.argv, n_total_obs)
	n_train = len(s_idx)
	kernel = pyro.infer.mcmc.NUTS(modeling.model, jit_compile=True)
	mcmc = pyro.infer.MCMC(kernel, num_samples=5, warmup_steps=10)
	mcmc.run(n_samp, n_drug, s_idx, d_idx, const.PARAMS, obs=obs_train, n_obs=n_train)
	mcmc_samples1 = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
	final_sample = get_final_sample(mcmc_samples1, 5)
	kernel = pyro.infer.mcmc.NUTS(modeling.model, jit_compile=True)
	mcmc = pyro.infer.MCMC(kernel, num_samples=5, warmup_steps=10)
	mcmc.run(n_samp, n_drug, s_idx, d_idx, const.PARAMS, obs=obs_train, n_obs=n_train)
	mcmc_samples2 = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
	# get one set of 10 samples
	pyro.set_rng_seed(0)
	pyro.clear_param_store()
	print('random val: ' + str(np.random.randn()))
	n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train, obs_test = get_synthetic_data(sys.argv, n_total_obs)
	n_train = len(s_idx)
	kernel = pyro.infer.mcmc.NUTS(modeling.model, jit_compile=True)
	mcmc = pyro.infer.MCMC(kernel, num_samples=10, warmup_steps=10)
	mcmc.run(n_samp, n_drug, s_idx, d_idx, const.PARAMS, obs=obs_train, n_obs=n_train)
	mcmc_samples3 = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
	print('two sets of 5 samples:')
	print(mcmc_samples1['a'])
	print(mcmc_samples2['a'])
	print('final sample of set 1: ')
	print(final_sample)
	print('one set of 10 samples:')
	print(mcmc_samples3['a'])

def test_simple_thinning():
	helpers.check_args(sys.argv, 10)
	save_args(sys.argv)
	n_total_obs, n_mcmc, n_warmup, n_iter, directory = params_from_args(sys.argv)
	# get thinned samples
	pyro.set_rng_seed(0)
	pyro.clear_param_store()
	print('random val: ' + str(np.random.randn()))
	n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train, obs_test = get_synthetic_data(sys.argv, n_total_obs)
	mcmc_thinned = get_mcmc_samples_with_simple_thinning(n_samp, n_drug, s_idx, d_idx, const.PARAMS, obs_train, n_mcmc=2, n_warmup=10, thinning=10)
	# get un-thinned samples
	pyro.set_rng_seed(0)
	pyro.clear_param_store()
	print('random val: ' + str(np.random.randn()))
	n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train, obs_test = get_synthetic_data(sys.argv, n_total_obs)
	mcmc_no_thin = get_mcmc_samples(n_samp, n_drug, s_idx, d_idx, const.PARAMS, obs_train, n_mcmc=20, n_warmup=10, initial_params=None)	
	print('mcmc_thinned - a: ')
	print(mcmc_thinned['a'])
	print('mcmc_no_thin - a: ')
	print(mcmc_no_thin['a'])

def test_thinning():
	helpers.check_args(sys.argv, 10)
	save_args(sys.argv)
	n_total_obs, n_mcmc, n_warmup, n_iter, directory = params_from_args(sys.argv)
	# get thinned samples
	pyro.set_rng_seed(0)
	pyro.clear_param_store()
	print('random val: ' + str(np.random.randn()))
	n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train, obs_test = get_synthetic_data(sys.argv, n_total_obs)
	mcmc_thinned = get_mcmc_samples_with_thinning(n_samp, n_drug, s_idx, d_idx, const.PARAMS, obs_train, n_mcmc=2, n_warmup=10, thinning=10)
	# get un-thinned samples
	pyro.set_rng_seed(0)
	pyro.clear_param_store()
	print('random val: ' + str(np.random.randn()))
	n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train, obs_test = get_synthetic_data(sys.argv, n_total_obs)
	mcmc_no_thin = get_mcmc_samples(n_samp, n_drug, s_idx, d_idx, const.PARAMS, obs_train, n_mcmc=20, n_warmup=10)	
	print('mcmc_thinned: ')
	print(mcmc_thinned)
	print('mcmc_no_thin: ')
	print(mcmc_no_thin)
	print('mcmc_thinned - a: ')
	print(mcmc_thinned['a'])
	print('mcmc_no_thin - a: ')
	print(mcmc_no_thin['a'])

def test_random_seed_with_generate_samples():
	helpers.check_args(sys.argv, 10)
	save_args(sys.argv)
	n_total_obs, n_mcmc, n_warmup, n_iter, directory = params_from_args(sys.argv)
	pyro.set_rng_seed(0)
	pyro.clear_param_store()
	n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train1, obs_test1 = get_synthetic_data(sys.argv, n_total_obs)
	pyro.set_rng_seed(0)
	pyro.clear_param_store()
	n_samp, n_drug, s_idx, d_idx, s_test_idx, d_test_idx, obs_train2, obs_test2 = get_synthetic_data(sys.argv, n_total_obs)
	print('obs_test1: ')
	print(obs_test1[0])
	print('obs_test2: ')
	print(obs_test2[0])

if __name__ == "__main__":
    #main()
    main_real_data()
    #main_synth_data()