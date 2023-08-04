import numpy as np
import pandas as pd
import pickle
import pyro.util
import sys

def get_raw_args(args, n):
    if len(args) != n + 1:
        print('Error! Expected ' + str(n) + ' arguments but got ' + str(len(args)))
        return
    data_fn = args[1].split('=')[1]
    fold_fn = args[2].split('=')[1]
    write_dir = args[3].split('=')[1]
    source_name = args[4].split('=')[1]
    target_name = args[5].split('=')[1]
    split_seed = int(args[6].split('=')[1])
    holdout_frac = float(args[7].split('=')[1])
    return data_fn, fold_fn, write_dir, source_name, target_name, split_seed, holdout_frac

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

def split_from_fold(df, fold_fn, model_seed):
    # read in split (list of arrays) from fold
    # index into arrays with model_seed to get fold
    # return train, test based on fold
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
    fold_list = read_pickle(fold_fn)
    # list of sample ids to hold out
    samples = list(fold_list[split_seed])
    # get all pairs corresponding to a set of samples
    train_df = df.loc[~df.sample_id.isin(sample_ids)].drop_duplicates()
    train_pairs = list(zip(test_df['sample_id'], test_df['drug_id']))
    test_df = df.loc[df.sample_id.isin(sample_ids)].drop_duplicates()
    test_pairs = list(zip(test_df['sample_id'], test_df['drug_id']))
    return train_pairs, test_pairs

# returns the list of (sample, drug) pairs in the test set
def get_train_test_pairs(df, fold_fn, split_seed, holdout_frac):
    if int(fold_fn) == -1:
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

def pearson_correlation(vec1, vec2):
    pearson_corr = np.corrcoef(vec1, vec2)
    return pearson_corr[0, 1]

def eval_raw(source_df, source_col, target_test_df, target_col):
    d = source_df.merge(target_test_df, on=['sample_id', 'drug_id'], validate='one_to_one')
    assert len(d) == len(target_test_df)
    source = d[source_col].to_numpy()
    target = d[target_col].to_numpy()
    return pearson_correlation(source, target)

def main():
    # get args
    data_fn, fold_fn, write_dir, source_name, target_name, split_seed, holdout_frac = get_raw_args(sys.argv, 7)
    #source_col = 'log_' + source_name + '_published_auc_mean'
    #target_col = 'log_' + target_name + '_published_auc_mean'
    source_col = source_name + '_auc_overlap_mean'
    target_col = target_name + '_auc_overlap_mean'
    source_df, target_train_df, target_test_df = get_source_and_target(data_fn, fold_fn, source_col, target_col, split_seed, holdout_frac)
    print('SOURCE')
    print(source_df.head())
    print('TARGET TRAIN')
    print(target_train_df.head())
    print('TARGET TEST')
    print(target_test_df.head())
    # in other files:
    # zscore source_df and target_train_df
    # fit model
    # predict test results (note, these are still z-scored)
    #### using zscore mean from target_train_df, "inverse z_score" these predictions
    #### this should output a vector of test predictions
    # evaluate by computing correlation between predictions and target_test_df
    corr = eval_raw(source_df, source_col, target_test_df, target_col)
    print(corr)
    # TODO: save correlation to file
    # TODO: write params to file!!!

if __name__ == "__main__":
    main()