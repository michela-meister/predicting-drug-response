import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

from scipy.stats import norm, normaltest

NBINS = 20

def normality_test(x, alpha):
    k2, p = normaltest(x)
    s = 'p = ' + str(round(p, 6)) + ' --> '
    if p < alpha:  # null hypothesis: x comes from a normal distribution
        s += 'p < ' + str(alpha) + '\n'
        s += 'The null hypothesis (that data come from normal dist) CAN be rejected.'
    else:
        s += 'p >= ' + str(alpha) + '\n'
        s += 'The null hypothesis (that data come from normal dist) CANNOT be rejected.'
    return s, p

def create_plot(val, group, min_duration, write_fn):
    if group == 'combined':
        x = list(df[val])
    elif group == 'treatment':
        x = list(df.loc[~df.control][val])
    elif group == 'control':
        x = list(df.loc[df.control][val])
    else:
        print('Error! Value group must be one of: combined, treatment, control.')
    fig = plt.figure()
    x_axis = np.arange(np.min(x), np.max(x), 0.001)
    mean = np.mean(x)
    std = np.std(x)
    sns.histplot(x, stat='density', bins=NBINS)
    plt.plot(x_axis, norm.pdf(x_axis, loc=mean, scale=std), color='r')
    plt.title(val + ', ' + group + ', days >= ' + str(min_duration) + ', N = ' + str(len(x)))
    s_normality, p = normality_test(x, .05)
    s = 'mean: ' + str(round(mean, 3)) + ', std: ' + str(round(std, 3)) + '\n'
    s += s_normality
    fig.text(.5, -.05, s, ha='center')
    plt.savefig(write_fn, bbox_inches='tight')
    plt.clf()
    plt.close()

NUM_ARGS = 4
n = len(sys.argv)
if n != NUM_ARGS:
    print("Error: " + str(NUM_ARGS) + " arguments needed; only " + str(n) + " arguments given.")
read_fn = sys.argv[1].split("=")[1]
write_dir = sys.argv[2].split("=")[1]
min_duration = int(sys.argv[3].split("=")[1])

# read in dataframe
df = pd.read_csv(read_fn)
# compute sample mean
df = df.merge(df.groupby(['Sample', 'Drug'])['log(V_V0)'].mean().reset_index(name='log(V_V0)_sm'),
              on=['Sample', 'Drug'],
              validate='many_to_one')
# compute mean-centered measurements
df['log(V_V0)_cen'] = df['log(V_V0)'] - df['log(V_V0)_sm']
# add control column
df['control'] = df['Drug'] == 'Vehicle'

create_plot('log(V_V0)_cen', 'combined', min_duration, write_dir + '/combined.png')
create_plot('log(V_V0)_cen', 'treatment', min_duration, write_dir + '/treatment.png')
create_plot('log(V_V0)_cen', 'control', min_duration, write_dir + '/control.png')


