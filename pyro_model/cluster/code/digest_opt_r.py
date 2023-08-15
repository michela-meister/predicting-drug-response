import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
import seaborn as sns
import sys

import helpers

def get_diagonal_args(args, n):
    if len(args) != n + 1:
        print('Expected ' + str(n + 1) + ' arguments, but got ' + str(len(args)))
        return
    results_dir = args[1].split("=")[1]
    k_max = int(args[2].split("=")[1])
    s_max = int(args[3].split("=")[1])
    m_max = int(args[4].split("=")[1])
    return results_dir, k_max, s_max, m_max

def read_diagonal_results(eval_type, data_dir, k_max, s_max, m_max):
    arr = np.ones((k_max, s_max, m_max)) * -np.inf
    for i in range(1, k_max + 1):
        r = i
        for s in range(0, s_max):
            for m in range(0, m_max):
                results_dir = data_dir + '/' + str(k_max) + '/'  + str(r) + '/' + str(s) + '/' + str(m)
                if eval_type == 'train':
                    fn = results_dir + '/train.pkl'
                elif eval_type == 'test':
                    fn = results_dir + '/test.pkl'
                else:
                    print('Error! Eval type must be train or test')
                res = helpers.read_pickle(fn)
                # r, k are 1-indexed; s, m are 0-indexed
                arr[i-1, s, m] = res
    # check all entries filled in
    assert np.sum(np.sum(np.sum(arr == -np.inf))) == 0
    return arr

def select_diagonal_models(arr):
    assert len(arr.shape) == 3
    k_dim, s_dim, m_dim = arr.shape
    model_idx = np.ones((k_dim, s_dim)) * -np.inf
    for i in range(0, k_dim):
        for s in range(0, s_dim):
            model_idx[i, s] = np.argmax(arr[i, s, :])
    return model_idx.astype(int)

def index_into_diagonal_models(arr, idx):
    assert len(arr.shape) == 3
    k_dim, s_dim, m_dim = arr.shape
    res = np.ones((k_dim, s_dim)) * -np.inf
    for i in range(0, k_dim):
        k = i
        r = i
        for s in range(0, s_dim):
            res[i, s] = arr[i, s, idx[i, s]]
    assert np.sum(np.sum(res == -np.inf)) == 0
    return res

def compute_diagonal_mean(arr, model_idx):
    assert len(arr.shape) == 3
    k_dim, s_dim, m_dim = arr.shape
    res = index_into_diagonal_models(arr, model_idx)
    avg = np.mean(res, axis=1)
    assert (avg.shape[0] == k_dim)
    assert np.sum(avg == -np.inf) == 0
    return avg

def digest_diagonal_results(results_dir, k_max, s_max, m_max):
    train_arr = read_diagonal_results('train', results_dir, k_max, s_max, m_max)
    test_arr = read_diagonal_results('test', results_dir, k_max, s_max, m_max)
    model_idx = select_diagonal_models(train_arr)
    train_avg = compute_diagonal_mean(train_arr, model_idx)
    test_avg = compute_diagonal_mean(test_arr, model_idx)
    train_fn = results_dir + '/train_avg.pkl'
    test_fn = results_dir + '/test_avg.pkl'
    helpers.write_pickle(train_avg, train_fn)
    helpers.write_pickle(test_avg, test_fn)

def plot_diagonal_and_save_max(write_fn, max_fn, read_train, read_test, n_seeds, n_restarts):
    train_vals = helpers.read_pickle(read_train)
    test_vals = helpers.read_pickle(read_test)
    test_argmax = np.argmax(test_vals)
    max_test_val = test_vals[test_argmax]
    best_k = test_argmax + 1 # because test_vals are 0-indexed, but k values only start at 1...
    helpers.write_pickle(best_k, max_fn)
    assert train_vals.shape == test_vals.shape
    title = 'plotting the diagonal: source dim k = transfer dim r\n'
    title += 'ea. datapoint averages ' + str(n_seeds) + ' seeds, ea. w/ ' + str(n_restarts) \
    + ' model restarts (kept max value)'
    plt.title(title)
    plt.xlabel('dimension r=k')
    plt.ylabel('pearson correlation')
    x_axis = range(1, len(train_vals) + 1)
    plt.plot(x_axis, train_vals, 'b-o', label='train')
    plt.plot(x_axis, test_vals, 'r-o', label='test')
    plt.plot(best_k, max_test_val, 'co', label='max-test')
    plt.legend()
    plt.savefig(write_fn)

def main():
    results_dir, k_max, s_max, m_max = get_diagonal_args(sys.argv, 4)
    digest_diagonal_results(results_dir, k_max, s_max, m_max)
    train_fn = results_dir + '/train_avg.pkl'
    test_fn = results_dir + '/test_avg.pkl'
    diag_fn = results_dir + '/diagonal.png'
    max_fn = results_dir + '/best_k.pkl'
    plot_diagonal_and_save_max(diag_fn, max_fn, train_fn, test_fn, s_max, m_max)

if __name__ == "__main__":
    main()

