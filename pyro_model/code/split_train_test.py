import numpy as np
import pandas as pd
import pickle
import sys

FRACTION_TRAIN = .75

def candidate_split(df, ntrain):
    # Collapse Fulvestrant drugs
    f_drugs = {'Fulvestrant (200 mg/kg)': 'Fulvestrant', 'Fulvestrant (40 mg/kg)': 'Fulvestrant'}
    df['drug_collapsed'] = df['drug'].replace(f_drugs)
    # Split on (sample, collapsed drug) pairs. This is to ensure that we don't end up with (sample A, Fulv 40mg) in
    # the test set and (sample A, Fulv 200mg) in the training set, as an example.
    p = df[['sample', 'drug_collapsed']].drop_duplicates().reset_index(drop=True)
    p = p.reindex(np.random.permutation(p.index))
    train_collapsed = p[:ntrain]
    test_collapsed = p[ntrain:]
    tups = df[['sample', 'drug', 'drug_collapsed']].drop_duplicates().reset_index(drop=True)
    train_pairs = train_collapsed.merge(tups, on=['sample', 'drug_collapsed'], validate='one_to_many')
    test_pairs = test_collapsed.merge(tups, on=['sample', 'drug_collapsed'], validate='one_to_many')
    train_pairs = train_pairs[['sample', 'drug']].drop_duplicates()
    test_pairs = test_pairs[['sample', 'drug']].drop_duplicates()
    return train_pairs, test_pairs

# checks if all the test values in column col appear in the training set
def check_subset(split):
    train, test = split
    if ((set(test['sample']) <= set(train['sample'])) and (set(test['drug']) <= set(train['drug']))):
        return True
    return False

def check_drugs(split, df):
    train, test = split
    df_drugs = df.drug.unique()
    test_drugs = test.drug.unique()
    # check that each drug in dataset appears in test
    for drug in df_drugs:
        if drug not in test_drugs:
            return False
    return True

def split_pairs(df, frac_train):
    npairs = len(df[['sample', 'drug']].drop_duplicates())
    ntrain = int(np.floor(npairs * frac_train))
    while True:
        split = candidate_split(df, ntrain)
        if check_subset(split) and check_drugs(split, df):
            return split
        
def split_data_by_pairs(df, pairs, vol_name):
    pairs['sample_drug_pair'] = pairs[['sample', 'drug']].apply(tuple, axis=1)
    pairs = pairs.merge(df, 
                        on=['sample', 'drug', 'sample_drug_pair'], 
                        validate='one_to_many')
    return pairs[['sample', 'drug', vol_name + '_obs', 'MID_list']]

def split_data(df, vol_name, frac_train):
    train_pairs, test_pairs = split_pairs(df, frac_train)
    df['sample_drug_pair'] = df[['sample', 'drug']].apply(tuple, axis=1)
    train = split_data_by_pairs(df, train_pairs, vol_name)
    test = split_data_by_pairs(df, test_pairs, vol_name)
    return train, test

def validate_disjoint(train, test):
    train_pairs = train[['sample', 'drug']].apply(tuple, axis=1)
    test_pairs = test[['sample', 'drug']].apply(tuple, axis=1)
    assert set(train_pairs).isdisjoint(set(test_pairs))
    
def validate_subset(train, test, col):
    train_vals = train[col].unique()
    test_vals = test[col].unique()
    assert set(test_vals).issubset(set(train_vals))
    
def validate_length(train, test, df):
    assert len(train) + len(test) == len(df)
    
def validate_split(train, test, df):
    validate_length(train, test, df)
    validate_disjoint(train, test)
    validate_subset(train, test, 'sample')
    validate_subset(train, test, 'drug')

def group_observations(df, vol_name):
    vol_list = df.groupby(['sample', 'drug'])[vol_name].apply(list).reset_index(name = vol_name + '_obs')
    mid_list = df.groupby(['sample', 'drug'])['MID'].apply(list).reset_index(name = 'MID_list')
    sd = df[['sample', 'drug']].drop_duplicates()
    sd = sd.merge(vol_list, on=['sample', 'drug'], validate='one_to_one')
    sd = sd.merge(mid_list, on=['sample', 'drug'], validate='one_to_one')
    return sd

def index_dict(elems):
    d = {}
    for elt in elems:
        if elt not in d.keys():
            d[elt] = len(d.keys())
    return d

# TODO: move to split / save side, since want train and test formatted the same way!
def format_cols(df):
    df = df[['sample', 'drug', 'log(V_V0)_obs']]
    df = df.explode('log(V_V0)_obs').reset_index(drop=True)
    df = df.rename(columns={'log(V_V0)_obs': 'log(V_V0)'})
    return df

def format_for_model(df, sample_dict, drug_dict):
    df = format_cols(df)
    df['s_idx'] = df['sample'].map(sample_dict)
    df['d_idx'] = df['drug'].map(drug_dict)
    return df

def write_to_pickle(obj, fn):
    with open(fn, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

NUM_ARGS = 3
n = len(sys.argv)
if n != NUM_ARGS:
    print("Error: " + str(NUM_ARGS) + " arguments needed; only " + str(n) + " arguments given.")
read_fn = sys.argv[1].split("=")[1]
write_dir = sys.argv[2].split("=")[1]

df = pd.read_csv(read_fn)
df = df[['Sample', 'Drug', 'log(V_V0)', 'MID']]
# map columns
df = df.rename(columns={'Sample': 'sample', 'Drug': 'drug'})
vol_name = 'log(V_V0)'
df = group_observations(df, vol_name)
train, test = split_data(df, vol_name, FRACTION_TRAIN)
validate_split(train, test, df)
# format train, test for model
sample_dict = index_dict(list(df['sample'].unique()))
drug_dict = index_dict(list(df['drug'].unique()))
train = format_for_model(train, sample_dict, drug_dict)
test = format_for_model(test, sample_dict, drug_dict)
train.to_pickle(write_dir + '/train.pkl')
test.to_pickle(write_dir + '/test.pkl')
write_to_pickle(sample_dict, write_dir + '/sample_dict.pkl')
write_to_pickle(drug_dict, write_dir + '/drug_dict.pkl')

