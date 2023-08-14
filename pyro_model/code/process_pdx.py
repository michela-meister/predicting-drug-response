import numpy as np
import pandas as pd

import helpers

BASE_DIR = 'results/2023-08-13_clust/pdx_expt'

results = []

for split_seed in list(range(10)):
	transfer_dir = BASE_DIR + '/transfer' + '/' + str(split_seed)
	transfer = helpers.read_pickle(transfer_dir + '/test.pkl') 
	raw_dir = BASE_DIR + '/raw' + '/' + str(split_seed)
	raw = helpers.read_pickle(raw_dir + '/test.pkl')
	results.append({'seed': split_seed, 'transfer': transfer, 'raw': raw})

df = pd.DataFrame(results)
df['dataset'] = 'pdo_to_pdx'
df.to_csv(BASE_DIR + '/analysis/results.csv', index=False)