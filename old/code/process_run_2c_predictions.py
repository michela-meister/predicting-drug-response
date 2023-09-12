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

transfer_results = []
raw_pub_results = []
for data_pair in DATA_PAIR_LIST:
	for split_seed in list(range(10)):
		source, target = get_source_and_target(data_pair)
		# get transfer data
		transfer_dir = BASE_DIR1 + '/' + data_pair + '/transfer' + '/' + str(split_seed)
		transfer_data = helpers.read_pickle(transfer_dir + '/test_predictions.pkl')
		pred_list = transfer_data['predictions']
		sample_id_list = transfer_data['sample_ids']
		drug_id_list = transfer_data['drug_id']
		# iterate through transfer_data, adding to dictionary
		for i in range(0, len(pred_list)):
			sample_id = sample_id_list[i]
			drug_id = drug_id_list[i]
			prediction = pred_list[i]
			transfer_results.append({'source': source, 'target': target, 'seed': split_seed, 'sample_id': sample_id, 'drug_id': drug_id, 'transfer_pred': prediction})
		# read in raw published data
		raw_dir = BASE_DIR1 + '/' + data_pair + '/raw' + '/' + str(split_seed)
		raw_data = helpers.read_pickle(raw_dir + '/test_predictions.pkl')
		pred_list = raw_data['predictions']
		sample_id_list = raw_data['sample_ids']
		drug_id_list = raw_data['drug_id']
		# iterate through transfer_data, adding to dictionary
		for i in range(0, len(pred_list)):
			sample_id = sample_id_list[i]
			drug_id = drug_id_list[i]
			prediction = pred_list[i]
			raw_pub_results.append({'source': source, 'target': target, 'seed': split_seed, 'sample_id': sample_id, 'drug_id': drug_id, 'raw_pub_pred': prediction})
df_transfer = pd.DataFrame(transfer_results)
df_raw_pub = pd.DataFrame(raw_pub_results)

raw_overlap_results = []
for data_pair in DATA_PAIR_LIST:
	for split_seed in list(range(10)): 
		source, target = get_source_and_target(data_pair)
		# read in raw auc overlap data
		raw_dir = BASE_DIR2 + '/' + data_pair + '/raw' + '/' + str(split_seed)
		raw_data = helpers.read_pickle(raw_dir + '/test_predictions.pkl')
		pred_list = raw_data['predictions']
		sample_id_list = raw_data['sample_ids']
		drug_id_list = raw_data['drug_id']
		# iterate through transfer_data, adding to dictionary
		for i in range(0, len(pred_list)):
			sample_id = sample_id_list[i]
			drug_id = drug_id_list[i]
			prediction = pred_list[i]
			raw_overlap_results.append({'source': source, 'target': target, 'seed': split_seed, 'sample_id': sample_id, 'drug_id': drug_id, 'raw_overlap_pred': prediction})
df_overlap_pub = pd.DataFrame(raw_overlap_results)		

df = df_transfer.merge(df_raw_pub, on=['source', 'target', 'seed', 'sample_id', 'drug_id'], validate='one_to_one')
df = df.merge(df_overlap_pub, on=['source', 'target', 'seed', 'sample_id', 'drug_id'], validate='one_to_one')
assert len(df) == len(df_transfer)
assert len(df) == len(df_raw_pub)
assert len(df) == len(df_overlap_pub)
print(len(df))

df.to_csv(WRITE_DIR + '/analysis/predictions.csv', index=False)