import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

NUM_ARGS = 3
n = len(sys.argv)
if n != NUM_ARGS:
    print("Error: " + str(NUM_ARGS) + " arguments needed; only " + str(n) + " arguments given.")
read_fn = sys.argv[1].split("=")[1]
write_dir = sys.argv[2].split("=")[1]

df = pd.read_csv(read_fn)
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
    fn = write_dir + '/' + str(mid) + '.png'
    plt.savefig(fn)