import numpy as np
import pandas as pd
import sys

FRACTION_TRAIN = .8

def candidate_split(df, ntrain):
    p = df[['sample', 'drug']].drop_duplicates().reset_index(drop=True)
    p = p.reindex(np.random.permutation(p.index))
    train = p[:ntrain]
    test = p[ntrain:]
    return train, test

# checks if all the test values in column col appear in the training set
def check_subset(split):
    train, test = split
    if ((set(test['sample']) <= set(train['sample'])) and (set(test['drug']) <= set(train['drug']))):
        return True
    return False

def split_pairs(df):
    npairs = len(df[['sample', 'drug']].drop_duplicates())
    ntrain = int(np.floor(npairs * FRACTION_TRAIN))
    while True:
        split = candidate_split(df, ntrain)
        if check_subset(split):
            return split
        
def split_data_by_pairs(df, pairs, vol_name):
    pairs['sample_drug_pair'] = pairs[['sample', 'drug']].apply(tuple, axis=1)
    pairs = pairs.merge(df, 
                        on=['sample', 'drug', 'sample_drug_pair'], 
                        validate='one_to_many')
    return pairs[['sample', 'drug', vol_name + '_obs']]

def split_data(df, vol_name):
    train_pairs, test_pairs = split_pairs(df)
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
    return df.groupby(['sample', 'drug'])[vol_name].apply(list).reset_index(name = vol_name + '_obs')


NUM_ARGS = 3
n = len(sys.argv)
if n != NUM_ARGS:
    print("Error: " + str(NUM_ARGS) + " arguments needed; only " + str(n) + " arguments given.")
read_fn = sys.argv[1].split("=")[1]
write_dir = sys.argv[2].split("=")[1]

df = pd.read_csv(read_fn)
df = df[['Sample', 'Drug', 'log(V_V0+1)']]
# map columns
df = df.rename(columns={'Sample': 'sample', 'Drug': 'drug'})
vol_name = 'log(V_V0+1)'
df = group_observations(df, vol_name)
train, test = split_data(df, vol_name)
validate_split(train, test, df)
train.to_pickle(write_dir + '/train.pkl')
test.to_pickle(write_dir + '/test.pkl')