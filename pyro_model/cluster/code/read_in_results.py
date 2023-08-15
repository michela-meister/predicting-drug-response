import numpy as np
import pickle

BASE_DIR = '/work/tansey/meisterm/results/2023-07-24/transfer_multi_eval'

def read_pickle(fn):
    with open(fn, 'rb') as f:
        v = pickle.load(f)
    return v

def write_to_pickle(obj, fn):
    with open(fn, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_rsq_arr(eval_type, data_dir, r_max, k_max, s_max, m_max):
    arr = np.ones((r_max, k_max, s_max, m_max)) * np.inf
    for r in range(1, r_max + 1):
        for k in range(1, k_max + 1):
            for s in range(1, s_max + 1):
                for m in range(1, m_max + 1):
                    fn = data_dir + '/' + str(r) + '/' + str(k) + '/' + str(s) + '/' + str(m) + '.pkl'
                    val = read_pickle(fn)
                    if eval_type == 'train':
                        res = val['rsq_train']
                    elif eval_type == 'test':
                        res = val['rsq_test']
                    else:
                        print('Error! Eval type must be train or test')
                    arr[r-1, k-1, s-1, m-1] = res
    assert np.sum(np.sum(np.sum(np.sum(arr == np.inf)))) == 0
    return arr

def model_selection(arr):
    assert len(arr.shape) == 4
    r_dim, k_dim, s_dim, m_dim = arr.shape
    model_idx = np.ones((r_dim, k_dim, s_dim)) * np.inf
    for r in range(0, r_dim):
        for k in range(0, k_dim):
            for s in range(0, s_dim):
                model_idx[r, k, s] = np.argmax(arr[r, k, s, :])
    return model_idx.astype(int)

def index_into_models(arr, idx):
    assert len(arr.shape) == 4
    r_dim, k_dim, s_dim, m_dim = arr.shape
    res = np.ones((r_dim, k_dim, s_dim)) * np.inf
    for r in range(0, r_dim):
        for k in range(0, k_dim):
            for s in range(0, s_dim):
                res[r, k, s] = arr[r, k, s, idx[r, k, s]]
    assert np.sum(np.sum(np.sum(res == np.inf))) == 0
    return res

def compute_mean(arr, model_idx):
    assert len(arr.shape) == 4
    r_dim, k_dim, s_dim, m_dim = arr.shape
    res = index_into_models(arr, model_idx)
    avg = np.mean(res, axis=2)
    assert (avg.shape[0] == r_max) and (avg.shape[1] == k_max)
    return avg

def save_results(results_dir, r_max, k_max, s_max, m_max):
	train_arr = read_rsq_arr('train', results_dir, r_max, k_max, s_max, m_max)
	test_arr = read_rsq_arr('test', results_dir, r_max, k_max, s_max, m_max)
	model_idx = model_selection(train_arr)
	train_avg = compute_mean(train_arr, model_idx)
	test_avg = compute_mean(test_arr, model_idx)
	train_fn = results_dir + '/train_avg.pkl'
	test_fn = results_dir + '/test_avg.pkl'
	write_to_pickle(train_avg, train_fn)
	write_to_pickle(test_avg, test_fn)

r_max = 10
k_max = 2
s_max = 10
m_max = 50
save_results(BASE_DIR + '/REP_GDSC', r_max, k_max, s_max, m_max)
#
save_results(BASE_DIR + '/REP_CTD2', r_max, k_max, s_max, m_max)
#
save_results(BASE_DIR + '/GDSC_REP', r_max, k_max, s_max, m_max)
#
save_results(BASE_DIR + '/GDSC_CTD2', r_max, k_max, s_max, m_max)
#
save_results(BASE_DIR + '/CTD2_REP', r_max, k_max, s_max, m_max)
#
OTHER_BASE = '/work/tansey/meisterm/results/2023-07-25/transfer_multi_eval'
save_results(OTHER_BASE + '/CTD2_GDSC', r_max, k_max, s_max, m_max)


#experiments = ['REP_GDSC', 'REP_CTD2', 'GDSC_REP', 'GDSC_CTD2', 'CTD2_REP']
# CTD2_GDSC
