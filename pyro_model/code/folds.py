import numpy as np
import pandas as pd
import sys

import helpers

NFOLDS = 20
SEED = 101

def get_fold_args(args, n):
    if len(args) != n + 1:
        print('Error! Expected ' + str(n + 1) + ' arguments but got ' + str(len(args)))
        return
    data_fn = args[1].split('=')[1]
    write_fn = args[2].split('=')[1]
    return data_fn, write_fn

def get_folds(seed, sample_ids, n_folds):
    np.random.seed(seed)
    perm = np.random.permutation(sample_ids)
    idx = np.round(np.linspace(0, len(sample_ids), n_folds + 1)).astype(int)
    folds = []
    for i in range(0, len(idx) - 1):
        start = idx[i]
        end = idx[i + 1]
        folds.append(perm[start:end])
    return folds

def main():
    # read in data
    data_fn, write_fn = get_fold_args(sys.argv, 2)
    df = pd.read_csv(data_fn)
    sample_ids = np.array(df.sample_id.unique())
    folds = get_folds(SEED, sample_ids, NFOLDS)
    helpers.write_pickle(folds, write_fn)

if __name__ == "__main__":
    main()