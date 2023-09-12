import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

import helpers

def histogram_r_sq(r_sq_fn, write_fn, use_real_data):
	prefix = 'Synthetic: '
	if use_real_data:
		prefix = 'Welm Data: '
	r_sq_list = np.loadtxt(r_sq_fn)
	fig = plt.figure()
	sns.histplot(r_sq_list)
	plt.title(prefix + 'r-squared values')
	plt.savefig(write_fn, bbox_inches='tight')
	plt.clf()
	plt.close()

def histogram_coverage(cov_fn, write_fn, use_real_data):
	prefix = 'Synthetic: '
	if use_real_data:
		prefix = 'Welm Data: '
	cov_list = np.loadtxt(cov_fn)
	fig = plt.figure()
	sns.histplot(cov_list)
	plt.title(prefix + 'coverage')
	plt.savefig(write_fn, bbox_inches='tight')
	plt.clf()
	plt.close()

# read in params
read_dir = sys.argv[1].split("=")[1]
n = int(sys.argv[2].split("=")[1])

# get use_real_data
inputs = helpers.read_pickle(read_dir + '/input_args.pkl')
use_real_data = inputs['use_real_data']

# for all directories
r_sq_list = []
cov_list = []
r_sq_fn = read_dir + '/r_squared_list.txt'
r_sq_plot_fn = read_dir + '/r_squared_plot.png'
cov_fn = read_dir + '/coverage_list.txt'
cov_plot_fn = read_dir + '/coverage_plot.png'
for i in range(1, n + 1):
	directory = read_dir + '/' + str(i)
	r_sq = np.loadtxt(directory + '/r_squared.txt')
	cov = np.loadtxt(directory + '/coverage.txt')
	r_sq_list.append(r_sq)
	cov_list.append(cov)
np.savetxt(r_sq_fn, r_sq_list)
np.savetxt(cov_fn, cov_list)
histogram_r_sq(r_sq_fn, r_sq_plot_fn, use_real_data)
histogram_coverage(cov_fn, cov_plot_fn, use_real_data)