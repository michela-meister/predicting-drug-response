import numpy as np
import pandas as pd
import sys

import helpers
import folds


def get_args(args, n):
    if len(args) != n + 1:
        print('Error! Expected ' + str(n + 1) + ' arguments but got ' + str(len(args)))
        return
    data_fn = args[1].split('=')[1]
    write_fn = args[2].split('=')[1]
    return data_fn, write_fn

def main():
    # read in data
    data_fn, write_fn = get_args(sys.argv, 2)
    df = pd.read_csv(data_fn)
    sample_ids = np.array(df.sample_id.unique())
    folds = []
    for sample_id in sample_ids:
        folds.append(np.array([sample_id]))
    helpers.write_pickle(folds, write_fn)

if __name__ == "__main__":
    main()