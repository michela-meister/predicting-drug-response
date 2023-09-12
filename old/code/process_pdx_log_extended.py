import numpy as np
import pandas as pd

import helpers

BASE_DIR = 'results/2023-08-14_clust/pdx_expt'
WRITE_DIR = 'results/2023-08-17/extended_pdx_expt'

results = []

for split_seed in list(range(18)):
	transfer_dir = BASE_DIR + '/transfer' + '/' + str(split_seed)
	transfer_pearson = helpers.read_pickle(transfer_dir + '/test.pkl')
	transfer_spearman = helpers.read_pickle(transfer_dir + '/test_spearman.pkl')  
	raw_dir = BASE_DIR + '/raw' + '/' + str(split_seed)
	raw_pearson = helpers.read_pickle(raw_dir + '/test.pkl')
	raw_spearman = helpers.read_pickle(raw_dir + '/test_spearman.pkl')
	results.append({'seed': split_seed, 'transfer_pearson': transfer_pearson, 'transfer_spearman': transfer_spearman, 'raw_pearson': raw_pearson, 'raw_spearman': raw_spearman})

df = pd.DataFrame(results)
df['dataset'] = 'pdo_to_pdx'
df.to_csv(WRITE_DIR + '/analysis/results.csv', index=False)