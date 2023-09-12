import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
import pickle
import seaborn as sns
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

def compute_mask_2d(k_max):
    arr = np.zeros((k_max, k_max)).astype(bool)
    for k in range(0, k_max):
        for r in range(0, k):
            arr[r, k] = True
    return arr

def check_for_missing_entries(arr, k_max, s_max, m_max):
    mask = compute_mask_2d(k_max)
    assert arr.shape == mask.shape
    dims = len(arr.shape)
    missing = arr[mask] == -np.inf
    for i in range(0, dims):
        missing = np.sum(missing)
    assert missing == 0

def read_results(eval_type, data_dir, k_max, s_max, m_max):
    arr = np.ones((k_max, k_max, s_max, m_max)) * -np.inf
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
    return arr

def select_models(arr):
    assert len(arr.shape) == 4
    k_dim, k_dim, s_dim, m_dim = arr.shape
    model_idx = np.ones((k_dim, k_dim, s_dim)) * -np.inf
    for k in range(0, k_dim):
        for r in range(0, k_dim):
            for s in range(0, s_dim):
                model_idx[r, k, s] = np.argmax(arr[r, k, s, :])
    return model_idx.astype(int)

def index_into_models(arr, idx):
    assert len(arr.shape) == 4
    k_dim, k_dim, s_dim, m_dim = arr.shape
    res = np.ones((k_dim, k_dim, s_dim)) * -np.inf
    for k in range(0, k_dim):
        for r in range(0, k_dim):
            for s in range(0, s_dim):
                res[r, k, s] = arr[r, k, s, idx[r, k, s]]
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

def save_figs_to_pdf(fig_list, write_fn):
    pdf = matplotlib.backends.backend_pdf.PdfPages(write_fn)
    for fig in fig_list:
        pdf.savefig(fig)
    pdf.close()

def get_masks(train, test):
    train_mask = train != -np.inf
    test_mask = test != -np.inf
    assert (train_mask == test_mask).all()
    return train_mask, test_mask

def get_min_value(train, test):
    train_mask, test_mask = get_masks(train, test)
    min_val = np.min(np.array(np.min(train[train_mask]), np.min(test[test_mask])))
    return min_val

def get_max_value(train, test):
    train_mask, test_mask = get_masks(train, test)
    max_val = np.max(np.array(np.max(train[train_mask]), np.max(test[test_mask])))
    return max_val

def plot_heatmap(read_train, write_train, read_test, write_test, write_all):
    # read in arrays from read_train, read_test
    train = read_pickle(read_train)
    test = read_pickle(read_test)
    # set upper and lower bounds for heatmap
    min_val = get_min_value(train, test)
    max_val = get_max_value(train, test)
    vmax = max_val + .01
    vmin = min_val - .01
    # for cells where r > k, set to vmin
    train_mask, test_mask = get_masks(train, test)
    train[~train_mask] = vmin
    test[~test_mask] = vmin
    # heatmap params
    assert train.shape == test.shape
    n_rows = train.shape[0]
    n_cols = train.shape[1]
    row_ticks = range(1, n_rows + 1)
    col_ticks = range(1, n_cols + 1)
    # train_fig
    train_fig = plt.figure()
    plt.title("train")
    ax = sns.heatmap(train, vmin=vmin, vmax=vmax, yticklabels=row_ticks, xticklabels=col_ticks)
    ax.set(xlabel='k', ylabel='r')
    plt.savefig(write_train)
    # test fig
    test_fig = plt.figure()
    plt.title("test")
    ax = sns.heatmap(test, vmin=vmin, vmax=vmax, yticklabels=row_ticks, xticklabels=col_ticks)
    ax.set(xlabel='k', ylabel='r')
    plt.savefig(write_test)
    # difference fig
    diff = train - test
    diff_fig = plt.figure()
    plt.title("train - test")
    ax = sns.heatmap(diff, yticklabels=row_ticks, xticklabels=col_ticks, cmap='crest')
    ax.set(xlabel='k', ylabel='r')
    # save all three figs to pdf
    save_figs_to_pdf([train_fig, test_fig, diff_fig], write_all)
    return

def plot_train_vs_test(train_vals, test_vals, title, x_axis, x_label, y_min, y_max):
    assert len(train_vals) == len(test_vals)
    fig = plt.figure()
    ax = plt.gca()
    ax.set_ylim([y_min, y_max])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('pearson correlation')
    plt.plot(x_axis, train_vals, 'b-o', label='train')
    plt.plot(x_axis, test_vals, 'r-o', label='test')
    plt.legend()
    return fig

def plot_fixed_r(write_fn, read_train, read_test, k_max, n_seeds, n_restarts):
    train = read_pickle(read_train)
    test = read_pickle(read_test)
    assert train.shape == test.shape
    # set upper and lower bounds for plots
    min_val = get_min_value(train, test)
    max_val = get_max_value(train, test)
    y_min = min_val - .02
    y_max = max_val + .02
    # create figures for each line plot
    fig_list = []
    for i in range(0, train.shape[0]):
        train_vals = train[i, i:]
        test_vals = test[i, i:]
        r = i + 1
        x_axis = range(r, k_max + 1)
        x_label = 'k (source dimension)'
        title = 'fixing transfer dimension r = ' + str(r) + '\n'
        title += 'ea. datapoint averages ' + str(n_seeds) + ' seeds, ea. w/ ' + str(n_restarts) \
        + ' model restarts (kept max value)'
        f = plot_train_vs_test(train_vals, test_vals, title, x_axis, x_label, y_min, y_max)
        fig_list.append(f)
    save_figs_to_pdf(fig_list, write_fn)

def plot_fixed_k(write_fn, read_train, read_test, n_seeds, n_restarts):
    train = read_pickle(read_train)
    test = read_pickle(read_test)
    assert train.shape == test.shape
    # set upper and lower bounds for plots
    min_val = get_min_value(train, test)
    max_val = get_max_value(train, test)
    y_min = min_val - .02
    y_max = max_val + .02
    # create figures for each line plot
    fig_list = []
    for j in range(0, train.shape[1]):
        train_vals = train[0:j+1, j]
        test_vals = test[0:j+1, j]
        k = j + 1
        x_axis = range(1, k + 1)
        x_label = 'r (transfer dimension)'
        title = 'fixing source dimension k = ' + str(k) + '\n'
        title += 'ea. datapoint averages ' + str(n_seeds) + ' seeds, ea. w/ ' + str(n_restarts) \
        + ' model restarts (kept max value)'
        f = plot_train_vs_test(train_vals, test_vals, title, x_axis, x_label, y_min, y_max)
        fig_list.append(f)
        plt.close()
    save_figs_to_pdf(fig_list, write_fn)

def main():
    results_dir, k_max, s_max, m_max = get_args(sys.argv, 4)
    digest_results(results_dir, k_max, s_max, m_max)
    train_fn = results_dir + '/train_avg.pkl'
    test_fn = results_dir + '/test_avg.pkl'
    heat_train_fn = results_dir + '/heatmap_train.png'
    heat_test_fn = results_dir + '/heatmap_test.png'
    heat_all_fn = results_dir + '/heatmap_all.pdf'
    fix_r_fn = results_dir + '/fixed_r.pdf'
    fix_k_fn = results_dir + '/fixed_k.pdf'
    plot_heatmap(train_fn, heat_train_fn, test_fn, heat_test_fn, heat_all_fn)
    plot_fixed_r(fix_r_fn, train_fn, test_fn, k_max, s_max, m_max)
    plot_fixed_k(fix_k_fn, train_fn, test_fn, s_max, m_max)

if __name__ == "__main__":
    main()

