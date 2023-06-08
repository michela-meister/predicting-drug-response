import numpy as np
import pandas as pd
from scipy import interpolate
import sys

END_DAY = 22
MIN_VOL = 1.0
NUM_ARGS = 4

def get_start(df):
	start = df.loc[df.groupby('MID')['Day'].idxmin()]
	return start.rename(columns = {'Day': 'start', 'Volume': 'start_vol'})

def get_end(df):
	end = df.loc[df.groupby('MID')['Day'].idxmax()]
	return end.rename(columns = {'Day': 'end', 'Volume': 'end_vol'})

def get_log_volume(df):
    # clip end volumes so that all are non-zero
    df['end_vol'] = df['end_vol'].clip(lower=MIN_VOL)
    df['V_V0'] = df['end_vol'].div(df['start_vol'])
    df['log(V_V0)'] = np.log2(df['V_V0'])
    return df

def split_by_duration(df, end_day):
    n_mids = df.MID.nunique()
    short_mids = df.loc[df.end < end_day].MID.unique()
    short_df = df.loc[df.MID.isin(short_mids)]
    normal_mids = df.loc[df.end >= end_day].MID.unique()
    normal_df = df.loc[df.MID.isin(normal_mids)]
    assert set(short_mids).isdisjoint(set(normal_mids))
    assert len(short_mids) + len(normal_mids) == n_mids
    return short_mids, short_df, normal_mids, normal_df

def assign_single_value(df, mids, max_val):
    df['log(V_V0)'] = max_val
    df = df[['MID', 'Sample', 'Drug', 'log(V_V0)']].drop_duplicates()
    assert (len(df) == len(mids))
    assert (set(df.MID.unique()) >= set(mids))
    return df

def assign_end_value(df, mids, end_day):
    # only keep rows for the end_day
    df = df.loc[df.Day == end_day]
    df = df[['MID', 'Sample', 'Drug', 'log(V_V0)']].drop_duplicates()
    # ensure no mid's dropped
    assert (len(df == len(mids)))
    assert (set(df.MID.unique()) >= set(mids))
    return df

def get_max_end_value(df, end_day):
	n_mids = df.MID.nunique()
	df = df.loc[df.Day <= end_day]
	# group by MID to get latest day for each MID
	latest = df.loc[df.groupby('MID')['Day'].idxmax()]
	assert len(latest) == n_mids
	return latest['log(V_V0)'].max()

def set_duration(df, end_day):
    n_mids = df.MID.nunique()
    mids = df.MID.unique()
    # get max value through end_day
    max_val = get_max_end_value(df, end_day)
    # split data between mids with short durations and mids with durations ending after end_day
    short_mids, short_df, normal_mids, normal_df = split_by_duration(df, end_day)
    # assign max_val to mids with short durations
    short_df = assign_single_value(short_df.copy(deep=True), short_mids, max_val)
    # assign individual value at end_day to normal mids
    normal_df = assign_end_value(normal_df, normal_mids, end_day)
    # concatenate short_df and normal_df together
    final_df = pd.concat([short_df, normal_df]).reset_index(drop=True)
    assert len(final_df) == n_mids
    assert set(final_df.MID.unique()) >= set(mids)
    return final_df
    
def get_start_and_end(df):
    start = get_start(df)
    end = get_end(df)
    n_mids = df.MID.nunique()
    assert (len(start) == n_mids) and (len(end) == n_mids)
    cols = ['MID', 'Sample', 'Drug']
    df = df.merge(start, on=cols, validate='many_to_one')
    df = df.merge(end, on=cols, validate='many_to_one')
    return df

n = len(sys.argv)
if n != NUM_ARGS:
    print("Error: " + str(NUM_ARGS) + " arguments needed; only " + str(n) + " arguments given.")
read_fn = sys.argv[1].split("=")[1]
write_dir = sys.argv[2].split("=")[1]
end_day = int(sys.argv[3].split("=")[1])

df = pd.read_csv(read_fn)
d = get_start_and_end(df)
# check all mids start on day 1
assert (d.start == 1).all()
d = get_log_volume(d)
d = set_duration(d, END_DAY)
d.to_csv(write_dir + '/welm_pdx_clean_mid_volume.csv', index=False)