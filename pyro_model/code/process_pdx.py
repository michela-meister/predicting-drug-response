import numpy as np
import pandas as pd

import helpers

BASE_DIR = 'results/2023-08-13_clust/pdx_expt'
WRITE_DIR = 'results/2023-08-17/pdx_expt'

results = []

for split_seed in list(range(12)):
	transfer_dir = BASE_DIR + '/transfer' + '/' + str(split_seed)
	transfer_pearson = helpers.read_pickle(transfer_dir + '/test.pkl') 
	raw_dir = BASE_DIR + '/raw' + '/' + str(split_seed)
	raw_pearson = helpers.read_pickle(raw_dir + '/test.pkl')
	results.append({'seed': split_seed, 'transfer_pearson': transfer_pearson, 'raw_pearson': raw_pearson})

df = pd.DataFrame(results)
df['dataset'] = 'pdo_to_pdx'
df.to_csv(WRITE_DIR + '/analysis/results.csv', index=False)