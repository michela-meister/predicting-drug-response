import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import mannwhitneyu 

## Nature journal settings
plt.rcParams["font.family"] = "Arial"
colors = ["#E64B35FF", "#3C5488FF", "#00A087FF", "#4DBBD5FF", "#F39B7FFF", "#8491B4FF", "#91D1C2FF", "#DC0000FF", "#7E6148FF", "#B09C85FF"]
sns.set_palette(sns.color_palette(colors))

fn = 'results_data/experiment1.csv'
df = pd.read_csv(fn)
df["dataset_name"] = df["source"] + "_" + df["target"]

# List of source-target pairs
dataset_names = ["GDSC_REP", "CTD2_REP", "REP_CTD2", "GDSC_CTD2", "REP_GDSC", "CTD2_GDSC"] 

# PRISM dataset is referred to as "REP" in raw data; convert to "PRSIM" name
xtick_names = []
for pair in dataset_names:
    source, target = pair.split('_')
    xtick_source = source
    xtick_target = target
    if xtick_source == 'REP':
        xtick_source = 'PRISM'
    if xtick_target == 'REP':
        xtick_target = 'PRISM'
    xtick_names.append(xtick_source + ' to ' + xtick_target)
    
test_xtick_names = [r"GDSC $\rightarrow$ PRISM", r"CTD2 $\rightarrow$ PRISM", r"PRISM $\rightarrow$ CTD2", r"GDSC $\rightarrow$ CTD2", r"PRISM $\rightarrow$ GDSC", r"CTD2 $\rightarrow$ GDSC"]

previous = "raw_overlap"
ours = "transfer"
method_names = [previous, ours]

for dataset in dataset_names:
    source_data, target_data = dataset.split("_")
    sub_df = df[(df["source"] == source_data) & (df["target"] == target_data)]
    print(source_data, target_data, "p-value=", mannwhitneyu(sub_df[previous], sub_df[ours]).pvalue)

plt.rcParams.update({"font.size":14}) ## Set fontsize

## Make the box plots
comp_colors = ['tab:gray', 'tab:orange']
fig, ax = plt.subplots(figsize=(14,4))
stats = []
positions = []
curr_pos = 0

for dataset in dataset_names:
    source_data, target_data = dataset.split("_")
    mu_bmt = np.inf
    mu_base = np.inf
    for method in method_names:
        v = df[(df["source"] == source_data) & (df["target"] == target_data)][method]
        mu = np.mean(v)
        if method == 'transfer':
            mu_bmt = mu
        if method == 'raw_overlap':
            mu_base = mu
        #print('method: ' + method + ' target: ' + target_data + ' mean: ' + str(mu))
        stdv = np.std(v)
        stats.append({"med":mu,           ## This is the line drawn in the box plot
                      "q1":mu-stdv,       ## This is the lower extent of the box plot
                      "q3":mu+stdv,       ## This is the upper extent of the box plot
                      "whislo":np.min(v), ## This is the lower extent of the whiskers
                      "whishi":np.max(v)})## This is the upper extent of the whiskers
        ax.scatter(x=(curr_pos + np.random.uniform(-1, 1, size=len(v))), ## Randomly scatter the points horizontally
                   y=v, 
                   color='white', 
                   edgecolors="black", 
                   zorder=-1)  ## zorder=-1 places the points behind the boxes 
        positions.append(curr_pos)
        curr_pos += 6
    #print('RELATIVE: ' + str((mu_bmt - mu_base)/mu_base))
    curr_pos += 5

## Actual code to plot
bplot = ax.bxp(stats, 
               positions=positions, 
               widths=5, 
               showfliers=False, 
               medianprops=dict(linestyle='-', linewidth=2.5, color='black'), 
               patch_artist=True, 
               zorder=0)

## Go through the boxes and set color + transparency
for i, patch in enumerate(bplot['boxes']):
    patch.set_facecolor(comp_colors[i%len(method_names)])
    patch.set_alpha(0.5)

xtickpos = [ 0.5*(positions[0]+positions[1]), 0.5*(positions[2]+positions[3]), 0.5*(positions[4]+positions[5]),
           0.5*(positions[6]+positions[7]), 0.5*(positions[8]+positions[9]), 0.5*(positions[10]+positions[11])]

plt.xticks(xtickpos, test_xtick_names)
plt.ylabel("Pearson correlation")
plt.ylim(np.min(df[previous])-0.1, np.max(df[ours])+0.15)


## Plot p-values!
h = 0.01
p_values = ["$p=0.00018$", "$p=0.00018$", "$p=0.00018$", "$p=0.00361$", "$p=0.00018$", "$p=0.00018$"] 

for i, dataset in enumerate(dataset_names):
    source_data, target_data = dataset.split("_")
    sub_df = df[(df["source"] == source_data) & (df["target"] == target_data)]
    y = np.max(sub_df[ours])+0.02
    x1, x2 = positions[2*i]-1, positions[2*i + 1]+1
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, color='k')
    plt.text((x1+x2)*.5, y+h, p_values[i], ha='center', va='bottom',color='k', fontsize=12)

patch1 = mpatches.Patch(color=comp_colors[0], alpha=0.5, label='Source overlap')
patch2 = mpatches.Patch(color=comp_colors[1], alpha=0.5, label='BMT')

plt.legend(handles=[patch1, patch2], ncol=2, fontsize=13, loc='upper right')
plt.savefig('figure1.png', bbox_inches='tight')

