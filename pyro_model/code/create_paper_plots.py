import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import sys

assert len(sys.argv) == 3
read_fn = sys.argv[1].split("=")[1]
write_dir = sys.argv[2].split("=")[1]


LEGEND_DRUG = {'Navitoclax': 'bo-', 'Vehicle': 'ko-', 'Docetaxel': 'co-', 'Birinapant': 'c+-', 'RO4929097': 'mo-', 
    'Irinotecan': 'yo-', 'Birinapant + Irinotecan': 'go-', 'Fulvestrant (40 mg/kg)': 'ro-', 'Fulvestrant (200 mg/kg)': 'r+-'}

LEGEND_SAMPLE = {'HCI-027': 'bo-', 'HCI-015': 'ko-', 'HCI-001': 'co-', 'HCI-002': 'mo-', 'HCI-019': 'yo-', 'HCI-012': 'ro-',
       'HCI-023': 'go-'}

df = pd.read_csv(read_fn)

# assert that each mid is in exactly 1 excel sheet
assert (df[['MID', 'excel_sheet']].groupby('MID')['excel_sheet'].nunique() == 1).all()

# TODO: normalize volume by day 0
# for each MID, get day 0 volume
start = df.loc[df.groupby('MID')['Day'].idxmin()]
start = start.rename(columns = {'Volume': 'start_vol'})
start = start[['MID', 'start_vol']].drop_duplicates()
df = df.merge(start, on=['MID'], validate='many_to_one')
df['V_V0'] = df['Volume'].div(df['start_vol'])
# merge the above with the full dataframe
# compute V/V0
# switch to plotting V/V0

# get all excel sheets in list
sheets = list(df.excel_sheet.unique())
exceptions = ['7c top', '7c bottom']
fig_list = []
# plot mid data from each excel sheet
for sheet in sheets:
	mids = list(df.loc[df.excel_sheet == sheet].MID.unique())
	f = plt.figure()
	if sheet not in exceptions:
		# get sample for sheet
		assert df.loc[df.excel_sheet == sheet].Sample.nunique() == 1
		sample = list(df.loc[df.excel_sheet == sheet].Sample.unique())[0]
		for mid in mids:
			m = df.loc[df.MID == mid]
			assert m.Drug.nunique() == 1
			drug = list(m.Drug.unique())[0]
			plt.plot(m.Day, m.V_V0, LEGEND_DRUG[drug], label=drug)
		plt.title('V/V0, Sheet: ' + sheet + ', Sample: ' + sample)
	else:
		# get drug for sheet
		assert df.loc[df.excel_sheet == sheet].Drug.nunique() == 1
		drug = list(df.loc[df.excel_sheet == sheet].Drug.unique())[0]
		for mid in mids:
			m = df.loc[df.MID == mid]
			assert m.Sample.nunique() == 1
			sample = list(m.Sample.unique())[0]
			plt.plot(m.Day, m.V_V0, LEGEND_SAMPLE[sample], label=sample)
		plt.title('V/V0, Sheet: ' + sheet + ', Drug: ' + drug)
	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = dict(zip(labels, handles))
	plt.legend(by_label.values(), by_label.keys())
	fig_list.append(f)
	plt.close(f)
# save figures to pdf
p = matplotlib.backends.backend_pdf.PdfPages(write_dir + '/recreate_welm_plots.pdf')
for f in fig_list:
    p.savefig(f)
p.close()
