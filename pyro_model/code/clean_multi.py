import numpy as np
import pandas as pd

def list_to_indices(keys):
    n_keys = len(keys)
    values = range(n_keys)
    return dict(zip(keys, values))

df = pd.read_csv('~/Documents/research/tansey/msk_intern/pyro_model/data/rep-gdsc-ctd2.csv')
columns = ['Drug.Name', 'ccle', 'REP_auc_overlap', 'CTD2_auc_overlap', 'GDSC_auc_overlap']
df = df[columns]
drugs = df['Drug.Name'].unique()
samples = df['ccle'].unique()
# map drugs, samples to indices
drug_indices = list_to_indices(drugs)
sample_indices = list_to_indices(samples)
df['drug_id'] = df['Drug.Name'].replace(drug_indices)
df['sample_id'] = df['ccle'].replace(sample_indices)
# save dataset
df.to_csv('~/Documents/research/tansey/msk_intern/pyro_model/data/rep-gdsc-ctd2-clean.csv', index=False)