import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

from scipy.stats import norm, normaltest

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
    sns.histplot(x, stat='density')
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
df = pd.read_pickle(read_fn)
# convert list to rows of datapoints
df = df[['sample', 'drug', 'log(V_V0+1)_obs']].explode('log(V_V0+1)_obs')
df = df.rename(columns={'log(V_V0+1)_obs': 'log(V_V0+1)'})
# compute sample mean
df = df.merge(df.groupby(['sample', 'drug'])['log(V_V0+1)'].mean().reset_index(name='log(V_V0+1)_sm'),
              on=['sample', 'drug'],
              validate='many_to_one')
# compute mean-centered measurements
df['log(V_V0+1)_cen'] = df['log(V_V0+1)'] - df['log(V_V0+1)_sm']
# add control column
df['control'] = df['drug'] == 'Vehicle'

create_plot('log(V_V0+1)_cen', 'combined', min_duration, write_dir + '/combined.png')
create_plot('log(V_V0+1)_cen', 'treatment', min_duration, write_dir + '/treatment.png')
create_plot('log(V_V0+1)_cen', 'control', min_duration, write_dir + '/control.png')


