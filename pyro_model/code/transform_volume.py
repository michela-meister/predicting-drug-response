import numpy as np
import pandas as pd
from scipy import interpolate
import sys

END_DAY = 22
MIN_VOL = 1.0
NUM_ARGS = 4

def get_start_day(df):
	start_day = df.loc[df.groupby('MID')['Day'].idxmin()]
	return start_day.rename(columns = {'Day': 'start', 'Volume': 'V0'})

def get_last_day(df):
	last_day = df.loc[df.groupby('MID')['Day'].idxmax()]
	return last_day.rename(columns = {'Day': 'end', 'Volume': 'end_vol'})

def add_volume_columns(df, start_day):
	assert len(start_day) == df.MID.nunique()
	old_len = len(df)
	df = df.merge(start_day[['MID', 'start', 'V0']], on='MID', validate='many_to_one')
	assert old_len == len(df)
	# compute functions of volume
	df['V_V0'] = df['Volume'].div(df['V0'])
	df['log(V_V0)'] = np.log2(df['V_V0'])
	return df

def drop_short_duration_mids(df, start_day, end_day):
	# ensure all mids start at day 1
	assert (start_day.start == 1).all()
	last_day = get_last_day(df)
	mids_to_drop = last_day.loc[last_day.end < end_day].MID.unique()
	return df.loc[~df.MID.isin(mids_to_drop)]

def create_end_day(df, end_day):
	end_df = df.loc[df.Day == end_day]
	assert df.MID.nunique() == end_df.MID.nunique()
	assert end_df.Day.isin([end_day]).all()
	return end_df[['MID', 'Sample', 'Drug', 'Volume', 'V_V0', 'log(V_V0)']]

n = len(sys.argv)
if n != NUM_ARGS:
    print("Error: " + str(NUM_ARGS) + " arguments needed; only " + str(n) + " arguments given.")
read_fn = sys.argv[1].split("=")[1]
write_dir = sys.argv[2].split("=")[1]
end_day = int(sys.argv[3].split("=")[1])

df = pd.read_csv(read_fn)
df['Volume'] = df['Volume'].clip(lower=MIN_VOL)
start_day = get_start_day(df)
df = add_volume_columns(df, start_day)
df = drop_short_duration_mids(df, start_day, end_day)
end_df = create_end_day(df, end_day)
end_df.to_csv(write_dir + '/welm_pdx_clean_mid_volume.csv', index=False)