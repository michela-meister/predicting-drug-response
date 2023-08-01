import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import sys

def list_to_indices(keys):
    n_keys = len(keys)
    values = range(n_keys)
    return dict(zip(keys, values))

def pearson_correlation(means, test):
    pearson_corr = np.corrcoef(test, means)
    return pearson_corr[0, 1]

read_dir = sys.argv[1].split("=")[1]
write_dir = sys.argv[2].split("=")[1]

# read in dataset
df = pd.read_csv(read_dir + '/rep-gdsc-ctd2.csv')
cols = ['Drug.Name', 'ccle', 'REP_published_auc', 'CTD2_published_auc', 'GDSC_published_auc', 'REP_auc_overlap', 'CTD2_auc_overlap', 'GDSC_auc_overlap']
df = df[cols]
# drop rows with nans
df = df.dropna()
# map drugs, samples to indices
drugs = df['Drug.Name'].unique()
samples = df['ccle'].unique()
drug_indices = list_to_indices(drugs)
sample_indices = list_to_indices(samples)
df['drug_id'] = df['Drug.Name'].replace(drug_indices)
df['sample_id'] = df['ccle'].replace(sample_indices)

id_cols = ['Drug.Name', 'ccle', 'drug_id', 'sample_id'] 
data_cols = ['REP_published_auc', 'CTD2_published_auc', 'GDSC_published_auc', 'REP_auc_overlap', 'CTD2_auc_overlap', 'GDSC_auc_overlap']
# average values for each (sample, drug) pair
mean_cols = []
for col in data_cols:
    d = df.groupby(['sample_id', 'drug_id'])[col].mean().reset_index(name = col + '_mean')
    df = df.merge(d, on=['sample_id', 'drug_id'], validate='many_to_one')
    mean_cols.append(col + '_mean')
# only keep columns with means
df = df[id_cols + mean_cols]
# drop duplicates due to merge
df = df.drop_duplicates()

# take the log of each column
log_cols = []
for col in data_cols:
	mean_col = col + '_mean'
	log_col = 'log_' + mean_col
	df[log_col] = np.log2(df[mean_col])
	log_cols.append(log_col)

# standardize all data
for col in mean_cols + log_cols:
    df[col] = scipy.stats.zscore(df[col].to_numpy())

# keep columns with averages
df = df[id_cols + mean_cols + log_cols]
# save to csv
df.to_csv(write_dir + '/rep-gdsc-ctd2-mean-log.csv', index=False)
