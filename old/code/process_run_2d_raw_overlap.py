import numpy as np
import pandas as pd

import helpers

def get_source_and_target(data_pair):
	str_list = data_pair.split("_")
	return str_list[1], str_list[2]

BASE_DIR = 'results/2023-08-31/run_2d_raw_overlap'

DATA_PAIR_LIST = ['log_REP_GDSC', 'log_REP_CTD2', 'log_GDSC_CTD2', 'log_GDSC_REP', 'log_CTD2_REP', 'log_CTD2_GDSC']

DICT_MAP = {'log_REP_GDSC': 'log_GDSC', 'log_REP_CTD2': 'log_CTD2', 'log_GDSC_CTD2': 'log_CTD2', 'log_GDSC_REP': 'log_REP', 
'log_CTD2_REP': 'log_REP', 'log_CTD2_GDSC': 'log_GDSC'}

# read in transfer,
results = []

PERCENT_LIST = ["10", "20", "30", "40", "50", "60", "70", "80", "85", "90", "95"]
for data_pair in DATA_PAIR_LIST:
	for percent in PERCENT_LIST:
		for split_seed in list(range(10)):
			source, target = get_source_and_target(data_pair)
			raw_dir = BASE_DIR + '/' + data_pair + '/' + percent + '/raw' + '/' + str(split_seed)
			raw = helpers.read_pickle(raw_dir + '/test.pkl')
			results.append({'source': source, 'target': target, 'percent-heldout': percent, 'seed': split_seed, 'raw_overlap': raw})

df = pd.DataFrame(results)
df.to_csv('results/2023-08-31/run_2d_raw_overlap' + '/analysis/results.csv', index=False)


