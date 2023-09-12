import numpy as np
import pandas as pd

import helpers

def get_source_and_target(data_pair):
	str_list = data_pair.split("_")
	return str_list[1], str_list[2]

BASE_DIR1 = 'results/2023-08-10_clust/run_2c'
BASE_DIR2 = 'results/2023-08-14_clust/run_2c_raw_overlap'
WRITE_DIR = 'results/2023-08-17/run_2c'
DATA_PAIR_LIST = ['log_REP_GDSC', 'log_REP_CTD2', 'log_GDSC_CTD2', 'log_GDSC_REP', 'log_CTD2_REP', 'log_CTD2_GDSC']

results = []

for data_pair in DATA_PAIR_LIST:
	for split_seed in list(range(10)):
		transfer_dir = BASE_DIR1 + '/' + data_pair + '/transfer' + '/' + str(split_seed)
		transfer = helpers.read_pickle(transfer_dir + '/test.pkl') 
		raw_dir = BASE_DIR1 + '/' + data_pair + '/raw' + '/' + str(split_seed)
		raw = helpers.read_pickle(raw_dir + '/test.pkl')
		source, target = get_source_and_target(data_pair)
		results.append({'source': source, 'target': target, 'seed': split_seed, 'transfer': transfer, 'raw_published': raw})
df_main = pd.DataFrame(results)

results_overlap = []
for data_pair in DATA_PAIR_LIST:
	for split_seed in list(range(10)): 
		raw_dir = BASE_DIR2 + '/' + data_pair + '/raw' + '/' + str(split_seed)
		raw = helpers.read_pickle(raw_dir + '/test.pkl')
		source, target = get_source_and_target(data_pair)
		results_overlap.append({'source': source, 'target': target, 'seed': split_seed, 'raw_overlap': raw})
df_overlap = pd.DataFrame(results_overlap)

df = df_main.merge(df_overlap, on=['source', 'target', 'seed'])

print(len(df))

df.to_csv(WRITE_DIR + '/analysis/results.csv', index=False)