import numpy as np
import pandas as pd
import sys

def enumerate_mid_names(mid_names):
    mid_dict = {}
    for x in range(0, len(mid_names)):
        mid_dict[mid_names[x]] = x
    return mid_dict

NUM_ARGS = 3
n = len(sys.argv)
if n != NUM_ARGS:
    print("Error: " + str(NUM_ARGS) + " arguments needed; only " + str(n) + " arguments given.")
read_fn = sys.argv[1].split("=")[1]
write_dir = sys.argv[2].split("=")[1]

df = pd.read_csv(read_fn)
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
df_out.to_csv(write_dir + '/welm_pdx_clean_mid.csv', index=False)
# save df with excel_sheet names for recreating paper plots
cols1 = ['MID', 'Sample', 'Drug', 'Day', 'Volume', 'excel_sheet']
df_out1 = df[cols1]
df_out1.to_csv(write_dir + '/welm_pdx_clean_mid_w_excel_sheet.csv', index=False)

