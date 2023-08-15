import numpy as np
import pandas as pd
import scipy

CLIP_LOWER = .01
CLIP_UPPER = .99

def list_to_indices(keys):
    n_keys = len(keys)
    values = range(n_keys)
    return dict(zip(keys, values))

df = pd.read_csv('~/Documents/research/tansey/msk_intern/pyro_model/data/rep-gdsc-ctd2.csv')
columns = ['Drug.Name', 'ccle', 'REP_auc_overlap', 'CTD2_auc_overlap', 'GDSC_auc_overlap']
df = df[columns]
# take log of auc's
df['log_REP_auc_overlap'] = np.log2(df['REP_auc_overlap'])
df['log_GDSC_auc_overlap'] = np.log2(df['GDSC_auc_overlap'])
df['log_CTD2_auc_overlap'] = np.log2(df['CTD2_auc_overlap'])
# clip and take logit of auc's
df['clip_REP_auc_overlap'] = df['REP_auc_overlap'].clip(lower=CLIP_LOWER, upper=CLIP_UPPER)
df['clip_GDSC_auc_overlap'] = df['GDSC_auc_overlap'].clip(lower=CLIP_LOWER, upper=CLIP_UPPER)
df['clip_CTD2_auc_overlap'] = df['CTD2_auc_overlap'].clip(lower=CLIP_LOWER, upper=CLIP_UPPER)
df['logit_REP_auc_overlap'] = scipy.special.logit(df['clip_REP_auc_overlap'])
df['logit_GDSC_auc_overlap'] = scipy.special.logit(df['clip_GDSC_auc_overlap'])
df['logit_CTD2_auc_overlap'] = scipy.special.logit(df['clip_CTD2_auc_overlap'])
# map drugs, samples to indices
drugs = df['Drug.Name'].unique()
samples = df['ccle'].unique()
drug_indices = list_to_indices(drugs)
sample_indices = list_to_indices(samples)
df['drug_id'] = df['Drug.Name'].replace(drug_indices)
df['sample_id'] = df['ccle'].replace(sample_indices)
# save dataset
df.to_csv('~/Documents/research/tansey/msk_intern/pyro_model/data/rep-gdsc-ctd2-clean.csv', index=False)