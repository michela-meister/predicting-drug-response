import numpy as np
import pandas as pd

k = 10
nrows = 300
ncols = 100
col_vecs = np.random.randn(nrows, k)
row_vecs = np.random.randn(k, ncols)
M = np.matmul(col_vecs, row_vecs)
sample_id, drug_id = np.where(M != -np.inf)
synth = M[sample_id, drug_id]
data = {'sample_id': sample_id, 'drug_id': drug_id, 'log_synth_published_auc_mean': synth}
df = pd.DataFrame(data)
df.to_csv('data/synth.csv')