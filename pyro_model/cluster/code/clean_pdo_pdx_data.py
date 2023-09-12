import numpy as np
import pandas as pd

# This script cleans the PDO-PDX data.

# Assigns index by position in list
def list_to_indices(keys):
    n_keys = len(keys)
    values = range(n_keys)
    return dict(zip(keys, values))

# Dataset to read in
pdo1_fn = 'data/yaspo_pdo1.csv'
pdo2_fn = 'data/yaspo_pdo2.csv'
pdx_fn = 'data/yaspo_pdx.csv'

# Read in first PDO dataset
df1 = pd.read_csv(pdo1_fn)
df1 = df1[['Patient ID', 'PDO', 'Drug', 'LOG10 IC50(µM)', 'Response category']]
df1 = df1.rename(columns={'Patient ID': 'patient_id', 'LOG10 IC50(µM)': 'log10_ic50_(uM)',
                         'Response category': 'pdo_response_category', 'PDO': 'pdo', 'Drug': 'drug'})
df1 = df1.drop_duplicates()

# Read in second PDO dataset
df2 = pd.read_csv(pdo2_fn)
df2 = df2[['Patient ID', 'PDO', 'Drug', 'LOG IC50 (µM)', 'Response category']]
df2 = df2.rename(columns={'Patient ID': 'patient_id', 'LOG IC50 (µM)': 'log10_ic50_(uM)',
                         'Response category': 'pdo_response_category', 'PDO': 'pdo', 'Drug': 'drug'})
df2 = df2.drop_duplicates()

# Concatenate to create full PDO dataset
pdo_df = pd.concat([df1, df2], ignore_index=True).drop_duplicates()
assert len(pdo_df) == len(df1) + len(df2)

# Read in PDX dataset
xf = pd.read_csv(pdx_fn)

# Select PDX data on list of drugs
drugs = ['cetuximab', 'AZD8931', 'selumetinib', 'afatinib', 'Avastin',
       'regorafenib', 'nintedanib', 'BI 860585', 'oxaliplatin', 'irinotecan',
       '5-FU']
xf = xf[['PDX'] + drugs]
xf = xf.dropna()
n_pdx = xf.PDX.nunique()
assert len(xf) == n_pdx
n_drugs = len(drugs)

# Write columns for PDX data
xf_list = []
for drug in drugs:
    drug_data = {'PDX': xf['PDX'], 'Drug': drug, 'T_C': xf[drug]}
    xf_list.append(pd.DataFrame(drug_data))
pdx_df = pd.concat(xf_list)
assert len(pdx_df) == len(drugs) * n_pdx
pdx_df = pdx_df.rename(columns={'PDX': 'pdx', 'Drug': 'drug'})

# assigning responses based on categories from excel sheet
pdx_df.loc[(pdx_df['T_C'] >= 0) & (pdx_df['T_C'] <= 10), 'pdx_response_category'] = 'Strong response'
pdx_df.loc[(pdx_df['T_C'] >= 11) & (pdx_df['T_C'] <= 25), 'pdx_response_category'] = 'Moderate response'
pdx_df.loc[(pdx_df['T_C'] >= 26) & (pdx_df['T_C'] <= 50), 'pdx_response_category'] = 'Minor response'
pdx_df.loc[(pdx_df['T_C'] > 50), 'pdx_response_category'] = 'Resistant'

# map pdx to patient id
pdx_to_id = {}
for pdx in list(pdx_df.pdx.unique()):
    if pdx.endswith('_XEN'):
        patient_id = pdx[:-4]
        pdx_to_id[pdx] = patient_id
pdx_df['patient_id'] = pdx_df['pdx'].map(pdx_to_id)

# merge PDO and PDX datasets
df = pdo_df.merge(pdx_df, on=['patient_id', 'drug'])

## Select "rectangle" of patients and drugs where each patient is tested against every drug in the set
drug_set = set(['regorafenib', '5-FU', 'oxaliplatin', 'nintedanib', 'BI 860585', 'selumetinib',
 'AZD8931', 'afatinib', 'irinotecan'])
patient_list = []
for patient_id in list(df.patient_id.unique()):
    if set(df.loc[df.patient_id == patient_id].drug.unique()) == drug_set:
        patient_list.append(patient_id)
df = df.loc[df.patient_id.isin(patient_list)]
df = df.loc[df.drug.isin(drug_set)]

# Assign patient ids, drug ids
drugs = list(df.drug.unique())
samples = list(df.patient_id.unique())
drug_indices = list_to_indices(drugs)
sample_indices = list_to_indices(samples)
df['drug_id'] = df['drug'].map(drug_indices)
df['sample_id'] = df['patient_id'].map(sample_indices)

df.to_csv('data/yaspo_combined.csv', index=False)