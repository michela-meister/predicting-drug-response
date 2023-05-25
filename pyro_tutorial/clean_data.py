import numpy as np
import pandas as pd

# edit DATA to load in welm_pdx.csv
DATA = '../welm/data/welm_pdx.csv'
df = pd.read_csv(DATA)

# Collapse drug names
df['Drug'] = df['Drug'].str.strip()
df['Drug'] = df['Drug'].replace('vehicle', 'Vehicle')

# Remove data from the following figures...
# Extended Data Fig 1: This data is from testing the response of tumor samples to estrogen.
# 3h: This data is an experiment involving retreatment related to drug resistance.
# '7d mid right', vehicle lines only: Data duplicated in Figure 6
# 7e, vehicle & birinapant lines: These are repeated in 7c
# 8: No drug overlap with in-vitro drugs. And Fig 8 is related to real-time tests for a single patient.
# So Fig 8 is somewhat different from other figs.

# Remove all data from Extended Data Fig 1
extended_data_fig1_fn = '43018_2022_337_MOESM11_ESM.xlsx'
df = df.loc[df['source_file'] != extended_data_fig1_fn]

# Remove all data from Fig 8
fig8_fn = '43018_2022_337_MOESM10_ESM.xlsx'
df = df.loc[df['source_file'] != fig8_fn]

# Remove data from 3h left and 3h right
df = df.loc[~df['excel_sheet'].isin(['3h left', '3h right'])]

# Remove data from 6d sheets, vehicle lines only
sheet_6d = ['6d left', '6d mid left', '6d mid', '6d mid right']
df = df.loc[~((df['excel_sheet'].isin(sheet_6d)) & (df['Drug'] == 'Vehicle'))]

# Remove data from sheet '7d mid right', vehicle lines only
df = df.loc[~((df['excel_sheet'] == '7d mid right') & (df['Drug'] == 'Vehicle'))]

# Remove data from 7e, vehicle and birinapant lines only
sheet_7e = ['7e left', '7e mid', '7e right']
df = df.loc[~((df['excel_sheet'].isin(sheet_7e)) & (df['Drug'].isin(['Birinapant', 'Vehicle'])))]

cols = ['Sample', 'Drug', 'Replicate Number', 'Day', 'Tumor Volume mm3']
assert df.groupby(cols)['source_file'].nunique().max() == 1
assert df.groupby(cols)['excel_sheet'].nunique().max() == 1

# save data
df.to_csv('data/welm_pdx_clean.csv', index=False)