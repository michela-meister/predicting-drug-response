import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('data/welm_pdx_clean_mid.csv')

# assert that each MID has exactly one sample, one drug
assert (df.groupby('MID').Sample.nunique() == 1).all()
assert (df.groupby('MID').Drug.nunique() == 1).all()

for mid in df.MID.unique():
    plt.clf()
    m = df.loc[df.MID == mid]
    sample = m.Sample.unique()[0]
    drug = m.Drug.unique()[0]
    plt.title('MID: ' + str(mid) + ', Sample: ' + sample + ', Drug: ' + drug)
    plt.xlabel('Day')
    plt.ylabel('Volume')
    plt.plot(m.Day, m.Volume)
    fn = 'results/2023-05-25/volume_vs_day/' + str(mid) + '.png'
    plt.savefig(fn)