import numpy as np
import pandas as pd

def enumerate_mid_names(mid_names):
    mid_dict = {}
    for x in range(0, len(mid_names)):
        mid_dict[mid_names[x]] = x
    return mid_dict

df = pd.read_csv('data/welm_pdx_clean.csv')
cols = ['Sample', 'Drug', 'Replicate Number', 'excel_sheet']
estimated_mids = len(df[cols].drop_duplicates())

#Give each Sample-Drug-Replicate Number-excel_sheet a unique MID
old_len = len(df)
df = df.merge(df.groupby(['Sample', 'Drug', 'Replicate Number', 'excel_sheet']).apply(lambda x: x.name).reset_index(name='MID'), 
              on=['Sample', 'Drug', 'Replicate Number', 'excel_sheet'], 
              validate='many_to_one')
mid_names = df['MID'].unique()
mid_dict = enumerate_mid_names(df['MID'].unique())
df['MID'] = df['MID'].map(mid_dict)
assert df.MID.nunique() == estimated_mids
assert len(df) == old_len
# Rename and select columns
df = df.rename(columns = {'Tumor Volume mm3': 'Volume'})
cols = ['MID', 'Sample', 'Drug', 'Day', 'Volume']
df_out = df[cols]
# Save file
df_out.to_csv('data/welm_pdx_clean_mid.csv', index=False)
