import numpy as np
import pandas as pd

import helpers

BASE_DIR = 'results/2023-08-13_clust/pdx_expt'
WRITE_DIR = 'results/2023-08-17/pdx_expt'


transfer_results = []
raw_results = []
for split_seed in list(range(12)):
	# get transfer data
	transfer_dir = BASE_DIR + '/transfer' + '/' + str(split_seed)
	transfer_data = helpers.read_pickle(transfer_dir + '/test_predictions.pkl')
	pred_list = transfer_data['predictions']
	sample_id_list = transfer_data['sample_ids']
	drug_id_list = transfer_data['drug_id']
	# iterate through transfer_data, adding to dictionary
	for i in range(0, len(pred_list)):
		sample_id = sample_id_list[i]
		drug_id = drug_id_list[i]
		prediction = pred_list[i]
		transfer_results.append({'seed': split_seed, 'sample_id': sample_id, 'drug_id': drug_id, 'transfer_pred': prediction})
	# read in raw published data
	raw_dir = BASE_DIR + '/raw' + '/' + str(split_seed)
	raw_data = helpers.read_pickle(raw_dir + '/test_predictions.pkl')
	pred_list = raw_data['predictions']
	sample_id_list = raw_data['sample_ids']
	drug_id_list = raw_data['drug_id']
	# iterate through transfer_data, adding to dictionary
	for i in range(0, len(pred_list)):
		sample_id = sample_id_list[i]
		drug_id = drug_id_list[i]
		prediction = pred_list[i]
		raw_results.append({'seed': split_seed, 'sample_id': sample_id, 'drug_id': drug_id, 'raw_pred': prediction})
df_transfer = pd.DataFrame(transfer_results)
df_raw = pd.DataFrame(raw_results)

df = df_transfer.merge(df_raw, on=['seed', 'sample_id', 'drug_id'], validate='one_to_one')

df.to_csv(WRITE_DIR + '/analysis/predictions.csv', index=False)