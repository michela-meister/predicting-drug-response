import numpy as np
import pandas as pd
import pickle
import pyro.util
import sys

def write_pickle(obj, fn):
    with open(fn, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(fn):
    with open(fn, 'rb') as handle:
        obj = pickle.load(handle)
    return obj

def get_unique_pairs(df, col1, col2):
    a = df[[col1, col2]].drop_duplicates()
    pairs = list(zip(a[col1], a[col2]))
    assert len(pairs) == len(set(pairs))
    return pairs

def get_train_test_indices(pairs, n_train):
    n_pairs = len(pairs)
    idx = np.random.permutation(n_pairs)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    assert set(train_idx).isdisjoint(set(test_idx))
    return train_idx, test_idx

def index_pairs(pairs, indices):
    p = np.array(pairs)
    idx = np.array(indices)
    return list(map(tuple, p[idx]))

def split_by_pairs(df, col1, col2, train_pairs, test_pairs):
    df['pair'] = list(zip(df[col1], df[col2]))
    train_df = df.loc[df['pair'].isin(train_pairs)]
    test_df = df.loc[df['pair'].isin(test_pairs)]
    assert set(train_df['pair']).isdisjoint(set(test_df['pair']))
    return train_df, test_df

def split_train_test(df):
    col1 = 'sample_id'
    col2 = 'drug_id'
    # get unique sample-drug pairs
    pairs = get_unique_pairs(df, col1, col2)
    n_train = int(np.ceil(len(pairs) * TRAIN_FRAC))
    train_idx, test_idx = get_train_test_indices(pairs, n_train)
    train_pairs = index_pairs(pairs, train_idx)
    test_pairs = index_pairs(pairs, test_idx)
    return split_by_pairs(df, col1, col2, train_pairs, test_pairs)

def random_split(df, split_seed, holdout_frac):
    pyro.util.set_rng_seed(split_seed)
    # get unique sample-drug pairs
    pairs = get_unique_pairs(df, 'sample_id', 'drug_id')
    n_train = int(np.ceil(len(pairs) * (1 - holdout_frac)))
    train_idx, test_idx = get_train_test_indices(pairs, n_train)
    train_pairs = index_pairs(pairs, train_idx)
    test_pairs = index_pairs(pairs, test_idx)
    return train_pairs, test_pairs

# get all (sample_id, drug_id) pairs in dataframe df with sample_id in samples
def pairs_from_samples(df, sample_ids):
    test_rows = df.loc[df.sample_id.isin(sample_ids)]
    return test_rows['pair'].to_numpy()

def fold_split(df, fold_fn, split_seed):
    fold_list = helpers.read_pickle(fold_fn)
    # list of sample ids to hold out
    samples = list(fold_list[split_seed])
    # get all pairs corresponding to a set of samples
    train_df = df.loc[~df.sample_id.isin(samples)].drop_duplicates()
    train_pairs = list(zip(train_df['sample_id'], train_df['drug_id']))
    test_df = df.loc[df.sample_id.isin(samples)].drop_duplicates()
    test_pairs = list(zip(test_df['sample_id'], test_df['drug_id']))
    assert set(train_pairs).isdisjoint(set(test_pairs))
    return train_pairs, test_pairs

# returns the list of (sample, drug) pairs in the test set
def get_train_test_pairs(df, fold_fn, split_seed, holdout_frac):
    if fold_fn == "":
        assert holdout_frac > 0 and holdout_frac <= 1
        return random_split(df, split_seed, holdout_frac)
    else:
        return fold_split(df, fold_fn, split_seed)

def split_train_test(df, fold_fn, split_seed, holdout_frac):
    # read in dataframe, and restrict to sample_id and drug_id
    train_pairs, test_pairs = get_train_test_pairs(df[['sample_id', 'drug_id']], fold_fn, split_seed, holdout_frac)
    train_df, test_df = split_by_pairs(df, 'sample_id', 'drug_id', train_pairs, test_pairs)
    return train_df, test_df

def get_source_and_target(data_fn, fold_fn, source_col, target_col, split_seed, holdout_frac):
    df = pd.read_csv(data_fn)
    source_df = df[['sample_id', 'drug_id', source_col]].drop_duplicates()
    train_df, test_df = split_train_test(df, fold_fn, split_seed, holdout_frac)
    target_train_df = train_df[['sample_id', 'drug_id', target_col]].drop_duplicates()
    target_test_df = test_df[['sample_id', 'drug_id', target_col]].drop_duplicates()
    return source_df, target_train_df, target_test_df

def get_sample_drug_ids(df):
    sd = df[['sample_id', 'drug_id']].drop_duplicates()
    # confirming that df has no duplicate pairs
    assert len(sd) == len(df)
    return sd

def pearson_correlation(vec1, vec2):
    pearson_corr = np.corrcoef(vec1, vec2)
    return pearson_corr[0, 1]