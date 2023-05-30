import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
import pandas as pd
import sys

NUM_ARGS = 3
YMIN = 0
YMAX = 1000

n = len(sys.argv)
if n != NUM_ARGS:
    print("Error: " + str(NUM_ARGS) + " arguments needed; only " + str(n) + " arguments given.")
read_fn = sys.argv[1].split("=")[1]
write_dir = sys.argv[2].split("=")[1]

df = pd.read_csv(read_fn)
# assert that each MID has exactly one sample, one drug
assert (df.groupby('MID').Sample.nunique() == 1).all()
assert (df.groupby('MID').Drug.nunique() == 1).all()
# create list of figures plotting mids growth curves
fig_list = []
for mid in df.MID.unique():
    f = plt.figure()
    m = df.loc[df.MID == mid]
    sample = m.Sample.unique()[0]
    drug = m.Drug.unique()[0]
    plt.title('MID: ' + str(mid) + ', Sample: ' + sample + ', Drug: ' + drug)
    plt.xlabel('Day')
    plt.ylabel('Volume mm^3')
    ax = plt.gca()
    ax.set_ylim([YMIN, YMAX])
    plt.plot(m.Day, m.Volume)
    fig_list.append(f)
    plt.close(f)
# save all figures to one pdf
p = matplotlib.backends.backend_pdf.PdfPages(write_dir + '/volume_vs_day.pdf')
for f in fig_list:
    p.savefig(f)
p.close()