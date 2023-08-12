import numpy as np
import pandas as pd

import helpers

BASE_DIR = 'results/2023-08-10_clust/run_2c'
DATA_PAIR_LIST = ['log_REP_GDSC', 'log_REP_CTD2', 'log_GDSC_CTD2', 'log_GDSC_REP', 'log_CTD2_REP', 'log_CTD2_GDSC']

results = []

for data_pair in DATA_PAIR_LIST:
	for split_seed in list(range(10)):
		transfer_dir = BASE_DIR + '/' + data_pair + '/transfer' + '/' + str(split_seed)
		transfer = helpers.read_pickle(transfer_dir + '/test.pkl') 
		raw_dir = BASE_DIR + '/' + data_pair + '/raw' + '/' + str(split_seed)
		raw = helpers.read_pickle(raw_dir + '/test.pkl')
		results.append({'data_pair': data_pair, 'seed': split_seed, 'transfer': transfer, 'raw': raw})

df = pd.DataFrame(results)
df.to_csv(BASE_DIR + '/analysis/results.csv', index=False)