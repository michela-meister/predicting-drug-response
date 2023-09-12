import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import mannwhitneyu ## for those wonderful p-values!

## Nature journal settings
plt.rcParams["font.family"] = "Arial"
colors = ["#E64B35FF", "#3C5488FF", "#00A087FF", "#4DBBD5FF", "#F39B7FFF", "#8491B4FF", "#91D1C2FF", "#DC0000FF", "#7E6148FF", "#B09C85FF"]
sns.set_palette(sns.color_palette(colors))

names = ["REP", "GDSC", "CTD2"]

d = pd.read_csv('run_2d/analysis/results.csv')
raw_overlap = pd.read_csv('../2023-08-31/run_2d_raw_overlap/analysis/results.csv')
d = d.merge(raw_overlap, on=['source', 'target', 'percent-heldout', 'seed'], validate='one_to_one')
d['pair'] = d['source'] + '_' + d['target']

for pair in d['pair'].unique():
	source, target = pair.split('_')
	df = d.loc[(d.source == source) & (d.target == target)]

	trans_std_df = df[['percent-heldout', 'transfer']].groupby(['percent-heldout']).std().reset_index()
	trans_mean_df = df[['percent-heldout', 'transfer']].groupby(['percent-heldout']).mean().reset_index()

	targ_std_df = df[['percent-heldout', 'target_only']].groupby(['percent-heldout']).std().reset_index()
	targ_mean_df = df[['percent-heldout', 'target_only']].groupby(['percent-heldout']).mean().reset_index()

	raw_std_df = df[['percent-heldout', 'raw_overlap']].groupby(['percent-heldout']).std().reset_index()
	raw_mean_df = df[['percent-heldout', 'raw_overlap']].groupby(['percent-heldout']).mean().reset_index()

	# Plot
	plt.rcParams.update({"font.size":12}) ## Set fontsize

	plt.clf()
	fig, ax = plt.subplots(figsize=(7,4))

	offset = 0.15

	i=0
	plt.errorbar(x=(raw_mean_df["percent-heldout"] + (i-1)*offset ), y=raw_mean_df["raw_overlap"], yerr=raw_std_df["raw_overlap"], color=colors[i], fmt='-o', label='raw_overlap', zorder=0)
	plt.scatter(x=(raw_mean_df["percent-heldout"] + (i-1)*offset ), y=raw_mean_df["raw_overlap"], color=colors[i], zorder=2)

	i=1
	plt.errorbar(x=(trans_mean_df["percent-heldout"] + (i-1)*offset ), y=trans_mean_df["transfer"], yerr=trans_std_df["transfer"], color=colors[i], fmt='-o', label='transfer', zorder=0)
	plt.scatter(x=(trans_mean_df["percent-heldout"] + (i-1)*offset ), y=trans_mean_df["transfer"], color=colors[i], zorder=2)

	i=2
	plt.errorbar(x=(targ_mean_df["percent-heldout"] + (i-1)*offset ), y=targ_mean_df["target_only"], yerr=targ_std_df["target_only"], color=colors[i], fmt='-o', label='target_only', zorder=0)
	plt.scatter(x=(targ_mean_df["percent-heldout"] + (i-1)*offset ), y=targ_mean_df["target_only"], color=colors[i], zorder=2)


	plt.legend(fontsize=16)
	plt.xticks([10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95])
	plt.xlabel("percent-heldout")
	plt.ylabel("Pearson Correlation")
	plt.title("source: " + source + ", target: " + target)
	plt.savefig("plots/line-" + source + "-" + target + ".png", bbox_inches="tight")





