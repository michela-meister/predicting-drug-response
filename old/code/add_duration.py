import numpy as np
import pandas as pd
import sys

NUM_ARGS = 4
n = len(sys.argv)
if n != NUM_ARGS:
    print("Error: " + str(NUM_ARGS) + " arguments needed; only " + str(n) + " arguments given.")
read_fn = sys.argv[1].split("=")[1]
write_dir = sys.argv[2].split("=")[1]
min_duration = int(sys.argv[3].split("=")[1])

df = pd.read_csv(read_fn)

def add_duration(df):
    num_mids = df.MID.nunique()
    # create columns with earliest day + size
    start_day = df.loc[df.groupby('MID')['Day'].idxmin()]
    start_day = start_day.rename(columns = {'Day': 'start', 'Volume': 'start_vol'})
    # create columns with latest day + size
    end_day = df.loc[df.groupby('MID')['Day'].idxmax()]
    end_day = end_day.rename(columns = {'Day': 'end', 'Volume': 'end_vol'})
    d = df[['MID', 'Sample', 'Drug']].drop_duplicates()
    # verify that all frames have the same length
    assert len(start_day) == num_mids
    assert len(end_day) == num_mids
    assert len(d) == num_mids
    # merge frames and take difference to find duration
    d = d.merge(start_day[['MID', 'start', 'start_vol']], on='MID', validate='one_to_one')
    d = d.merge(end_day[['MID', 'end', 'end_vol']], on='MID', validate='one_to_one')
    assert len(d) == num_mids
    d['duration'] = d['end'] - d['start']
    return d[['MID', 'Sample', 'Drug', 'start_vol', 'end_vol', 'duration']]

df = add_duration(df)
# Select only values with duration at least min_duration
df = df.loc[df.duration >= min_duration]
# Compute function of volume
df['V_V0'] = df['end_vol'].div(df['start_vol'])
df['log(V_V0+1)'] = np.log2(df['V_V0'] + 1)
df.to_csv(write_dir + '/welm_pdx_clean_mid_volume.csv', index=False)