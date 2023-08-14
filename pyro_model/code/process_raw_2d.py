import numpy as np
import pandas as pd

import helpers

BASE_DIR = 'results/2023-08-10_clust/run_2d'

DATA_PAIR_LIST = ['log_REP_GDSC', 'log_REP_CTD2', 'log_GDSC_CTD2', 'log_GDSC_REP', 'log_CTD2_REP', 'log_CTD2_GDSC']

DICT_MAP = {'log_REP_GDSC': 'log_GDSC', 'log_REP_CTD2': 'log_CTD2', 'log_GDSC_CTD2': 'log_CTD2', 'log_GDSC_REP': 'log_REP', 
'log_CTD2_REP': 'log_REP', 'log_CTD2_GDSC': 'log_GDSC'}

PERCENT_LIST = ["20", "40", "60", "80"]

results = []

for data_pair in DATA_PAIR_LIST:
	for percent in PERCENT_LIST:
		for split_seed in list(range(10)):
			transfer_dir = BASE_DIR + '/' + data_pair + '/' + percent + '/transfer' + '/' + str(split_seed)
			transfer = helpers.read_pickle(transfer_dir + '/test.pkl') 
			raw_dir = BASE_DIR + '/' + data_pair + '/' + percent + '/raw' + '/' + str(split_seed)
			raw = helpers.read_pickle(raw_dir + '/test.pkl')
			target_name = DICT_MAP[data_pair]
			target_only_dir = BASE_DIR + '/' + target_name + '/' + percent + '/target_only' + '/' + str(split_seed)
			target_only = helpers.read_pickle(target_only_dir + '/test.pkl')
			results.append({'dataset': data_pair, 'percent-heldout': percent, 'seed': split_seed, 'transfer': transfer, 'raw': raw, 'target_only': target_only})


# Get later runs
BASE_DIR = 'results/2023-08-11_clust/run_2d'
PERCENT_LIST = ["10", "85", "90", "95"]

for data_pair in DATA_PAIR_LIST:
	for percent in PERCENT_LIST:
		for split_seed in list(range(10)):
			transfer_dir = BASE_DIR + '/' + data_pair + '/' + percent + '/transfer' + '/' + str(split_seed)
			transfer = helpers.read_pickle(transfer_dir + '/test.pkl') 
			raw_dir = BASE_DIR + '/' + data_pair + '/' + percent + '/raw' + '/' + str(split_seed)
			raw = helpers.read_pickle(raw_dir + '/test.pkl')
			target_name = DICT_MAP[data_pair]
			target_only_dir = BASE_DIR + '/' + target_name + '/' + percent + '/target_only' + '/' + str(split_seed)
			target_only = helpers.read_pickle(target_only_dir + '/test.pkl')
			results.append({'dataset': data_pair, 'percent-heldout': percent, 'seed': split_seed, 'transfer': transfer, 'raw': raw, 'target_only': target_only})

# And even later runs...
BASE_DIR = 'results/2023-08-12_clust/run_2d'
PERCENT_LIST = ["30", "50", "70"]

for data_pair in DATA_PAIR_LIST:
	for percent in PERCENT_LIST:
		for split_seed in list(range(10)):
			transfer_dir = BASE_DIR + '/' + data_pair + '/' + percent + '/transfer' + '/' + str(split_seed)
			transfer = helpers.read_pickle(transfer_dir + '/test.pkl') 
			raw_dir = BASE_DIR + '/' + data_pair + '/' + percent + '/raw' + '/' + str(split_seed)
			raw = helpers.read_pickle(raw_dir + '/test.pkl')
			target_name = DICT_MAP[data_pair]
			target_only_dir = BASE_DIR + '/' + target_name + '/' + percent + '/target_only' + '/' + str(split_seed)
			target_only = helpers.read_pickle(target_only_dir + '/test.pkl')
			results.append({'dataset': data_pair, 'percent-heldout': percent, 'seed': split_seed, 'transfer': transfer, 'raw': raw, 'target_only': target_only})


df = pd.DataFrame(results)
df.to_csv('results/2023-08-10_clust/run_2d' + '/analysis/results.csv', index=False)