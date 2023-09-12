import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import spearmanr

## Nature journal settings
plt.rcParams["font.family"] = "Arial"
colors = ["#E64B35FF", "#3C5488FF", "#00A087FF", "#4DBBD5FF", "#F39B7FFF", "#8491B4FF", "#91D1C2FF", "#DC0000FF", "#7E6148FF", "#B09C85FF"]
sns.set_palette(sns.color_palette(colors))

plt.rcParams.update({"font.size":18}) ## Set fontsize

df_fn = 'results/experiment2.csv'
df = pd.read_csv(df_fn)
df['frac-used'] = df['percent-used'] / 100.0

np.random.seed(100)
## Dataset info
dataset_names = ["GDSC_REP", "CTD2_REP", "REP_CTD2", "GDSC_CTD2", "REP_GDSC", "CTD2_GDSC"] 
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
    
variable_values = [5, 10, 15, 20]

## p-values
for dataset in dataset_names:
    sub_df = df[(df["pair"]==dataset) & df["percent-used"].isin(variable_values)]
    print(dataset, spearmanr(sub_df["percent-used"], sub_df["efficiency-gain"]))

curr_pos = 0

fig, ax = plt.subplots(figsize=(15,6))
stats = []
positions = []
for dataset in dataset_names:
    for var in variable_values:
        v = df[(df["pair"]==dataset)&(df["percent-used"]==var)]["efficiency-gain"]
        mu = np.mean(v)
        #print('dataset: ' + dataset + ' var: ' + str(var) + ' mean: ' + str(mu))
        stdv = np.std(v)
        stats.append({"med":mu, 
                      "q1":mu-stdv, 
                      "q3":mu+stdv, 
                      "whislo":np.min(v), 
                      "whishi":np.max(v), 
                      "label":str(var)})
        
        ax.scatter(x=(curr_pos + np.random.uniform(-0.75, 0.75, size=len(v))), 
                   y=v, 
                   color='white', 
                   edgecolors="black", 
                   zorder=-1)
        positions.append(curr_pos)
        curr_pos += 6
    curr_pos += 9

## Make boxplots
bplot = ax.bxp(stats, 
               positions=positions, 
               widths=4, 
               showfliers=False, 
               medianprops=dict(linestyle='-', linewidth=2.5, color='black'), 
               patch_artist=True, 
               zorder=0)
for i, patch in enumerate(bplot['boxes']):
    patch.set_facecolor(colors[i//4]) ## Use those nature colors
    patch.set_alpha(0.5)

plt.xlabel("Percentage of target dataset used in training")
plt.ylabel("Efficiency gain")
plt.ylim(np.min(df['efficiency-gain'])-0.1, np.max(df['efficiency-gain'])+0.55)

## P-values + spearman rho
p_values = [r"$\rho_s = -0.96$, $p=10^{-21}$", r"$\rho_s = -0.95$, $p=10^{-20}$", r"$\rho_s = -0.89$, $p=10^{-13}$", r"$\rho_s = -0.64$, $p=10^{-5}$", r"$\rho_s = -0.94$, $p=10^{-18}$", r"$\rho_s = -0.93$, $p=10^{-16}$",] 

for i, dataset in enumerate(dataset_names):
    h = 0.02
    sub_df = df[(df["pair"]==dataset)]
    y = np.max(sub_df["efficiency-gain"])+0.1
    x1, x2 = positions[4*i]-1, positions[4*i + 3]+1
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, color='k')
    plt.text((x1+x2)*.5, y+h, p_values[i], ha='center', va='bottom',color='k', fontsize=14)
    
patch1 = mpatches.Patch(color=colors[0], alpha=0.5, label=test_xtick_names[0])
patch2 = mpatches.Patch(color=colors[1], alpha=0.5, label=test_xtick_names[1])
patch3 = mpatches.Patch(color=colors[2], alpha=0.5, label=test_xtick_names[2])
patch4 = mpatches.Patch(color=colors[3], alpha=0.5, label=test_xtick_names[3])
patch5 = mpatches.Patch(color=colors[4], alpha=0.5, label=test_xtick_names[4])
patch6 = mpatches.Patch(color=colors[5], alpha=0.5, label=test_xtick_names[5])

plt.plot([-5, 187], [1, 1], color='k', linestyle='dashed', label='baseline')
b_line = Line2D([-5, 187], [1, 1], color='k', linestyle='dashed', label='baseline')

ax.legend(handles=[patch1, patch2, patch3, patch4, patch5, patch6, b_line], ncol=1, fontsize=13, loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("figure2.png", bbox_inches='tight')




