import numpy as np 
import pickle
import sys

def read_pickle(fn):
	with open(fn, 'rb') as handle:
		val = pickle.load(handle)
	return val

def get_args(args, n):
	if len(args) != n + 1:
		print('Expected ' + str(n + 1) + ' arguments, but got ' + str(len(args)))
	data_dir = args[1].split("=")[1]
	r_max = int(args[2].split("=")[1])
	k_max = int(args[3].split("=")[1])
	seed_max = int(args[4].split("=")[1])
	return data_dir, r_max, k_max, seed_max

def retrieve_rsq_results():
	# read in args
	data_dir, r_max, k_max, seed_max = get_args(sys.argv, 4)
	train_rsq_arr = np.ones((r_max, k_max)) * np.inf 
	test_rsq_arr = np.ones((r_max, k_max)) * np.inf
	# get starting directory name
	for r in range(1, r_max + 1):
		for k in range(1, k_max + 1):
			rsq_train = []
			rsq_test = []
			for seed in range(1, seed_max + 1):
				fn = data_dir + '/' + str(r) + '/' + str(k) + '/' + str(seed) + '.pkl'
				vals_dict = read_pickle(fn)
				assert r == vals_dict['r']
				assert k == vals_dict['k']
				assert seed == vals_dict['seed']
				rsq_train.append(vals_dict['rsq_train'])
				rsq_test.append(vals_dict['rsq_test'])
			train_rsq_arr[r-1, k-1] = np.mean(np.array(rsq_train))
			test_rsq_arr[r-1, k-1] = np.mean(np.array(rsq_test))
	assert np.sum(np.sum(train_rsq_arr == np.inf)) == 0
	assert np.sum(np.sum(test_rsq_arr == np.inf)) == 0
	np.savetxt(data_dir + '/train_rsq_arr.txt', train_rsq_arr)
	np.savetxt(data_dir + '/test_rsq_arr.txt', test_rsq_arr)

def main():
	retrieve_rsq_results()

if __name__ == "__main__":
	main()