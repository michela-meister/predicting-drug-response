import numpy as np 
import pickle
import sys

def get_args(args, n):
    if len(args) != n + 1:
        print('Expected ' + str(n + 1) + ' arguments, but got ' + str(len(args)))
        return
    results_dir = args[1].split("=")[1]
    k_max = int(args[2].split("=")[1])
    s_max = int(args[3].split("=")[1])
    m_max = int(args[4].split("=")[1])
    return results_dir, k_max, s_max, m_max

def read_pickle(fn):
    with open(fn, 'rb') as f:
        v = pickle.load(f)
    return v

def write_to_pickle(obj, fn):
    with open(fn, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def compute_mask_4d(k_max, s_max, m_max):
    arr = np.zeros((k_max, k_max, s_max, m_max)).astype(bool)
    for k in range(0, k_max):
        for r in range(0, k):
            for s in range(0, s_max):
                for m in range(0, m_max):
                    arr[r, k, s, m] = True
    return arr

def compute_mask_3d(k_max, s_max):
    arr = np.zeros((k_max, k_max, s_max)).astype(bool)
    for k in range(0, k_max):
        for r in range(0, k):
            for s in range(0, s_max):
                arr[r, k, s] = True
    return arr

def compute_mask_2d(k_max):
    arr = np.zeros((k_max, k_max)).astype(bool)
    for k in range(0, k_max):
        for r in range(0, k):
            arr[r, k] = True
    return arr

def compute_mask(arr, k_max, s_max, m_max):
    dims = len(arr.shape)
    if dims == 4:
        return compute_mask_4d(k_max, s_max, m_max)
    elif dims == 3:
        return compute_mask_3d(k_max, s_max)
    elif dims == 2:
        return compute_mask_2d(k_max)
    else:
        print('Error! dims should be in {2, 3, 4}.')

def check_for_missing_entries(arr, k_max, s_max, m_max):
    mask = compute_mask(arr, k_max, s_max, m_max)
    assert arr.shape == mask.shape
    dims = len(arr.shape)
    missing = arr[mask] == -1
    for i in range(0, dims):
        missing = np.sum(missing)
    assert missing == 0

def read_results(eval_type, data_dir, k_max, s_max, m_max):
    arr = np.ones((k_max, k_max, s_max, m_max)) * -1
    for k in range(1, k_max + 1):
        for r in range(1, k + 1):
            for s in range(1, s_max + 1):
                for m in range(1, m_max + 1):
                    fn = data_dir + '/' + str(r) + '/' + str(k) + '/' + str(s) + '/' + str(m) + '.pkl'
                    val = read_pickle(fn)
                    if eval_type == 'train':
                        res = val['corr_train']
                    elif eval_type == 'test':
                        res = val['corr_test']
                    else:
                        print('Error! Eval type must be train or test')
                    arr[r-1, k-1, s-1, m-1] = res
    check_for_missing_entries(arr, k_max, s_max, m_max)
    return arr

def select_models(arr):
    assert len(arr.shape) == 4
    k_dim, k_dim, s_dim, m_dim = arr.shape
    model_idx = np.ones((k_dim, k_dim, s_dim)) * -1
    for k in range(0, k_dim):
        for r in range(0, k):
            for s in range(0, s_dim):
                model_idx[r, k, s] = np.argmax(arr[r, k, s, :])
    check_for_missing_entries(model_idx, k_dim, s_dim, m_dim)
    return model_idx.astype(int)

def index_into_models(arr, idx):
    assert len(arr.shape) == 4
    k_dim, k_dim, s_dim, m_dim = arr.shape
    res = np.ones((k_dim, k_dim, s_dim)) * -1
    for k in range(0, k_dim):
        for r in range(0, k):
            for s in range(0, s_dim):
                res[r, k, s] = arr[r, k, s, idx[r, k, s]]
    check_for_missing_entries(res, k_dim, s_dim, m_dim)
    return res

def compute_mean(arr, model_idx):
    assert len(arr.shape) == 4
    k_dim, k_dim, s_dim, m_dim = arr.shape
    res = index_into_models(arr, model_idx)
    avg = np.mean(res, axis=2)
    assert (avg.shape[0] == k_dim) and (avg.shape[1] == k_dim)
    check_for_missing_entries(avg, k_dim, s_dim, m_dim)
    return avg

def digest_results(results_dir, k_max, s_max, m_max):
    train_arr = read_results('train', results_dir, k_max, s_max, m_max)
    test_arr = read_results('test', results_dir, k_max, s_max, m_max)
    model_idx = select_models(train_arr)
    train_avg = compute_mean(train_arr, model_idx)
    test_avg = compute_mean(test_arr, model_idx)
    train_fn = results_dir + '/train_avg.pkl'
    test_fn = results_dir + '/test_avg.pkl'
    write_to_pickle(train_avg, train_fn)
    write_to_pickle(test_avg, test_fn)

def main():
    results_dir, k_max, s_max, m_max = get_args(sys.argv, 4)
    digest_results(results_dir, k_max, s_max, m_max)

if __name__ == "__main__":
    main()

