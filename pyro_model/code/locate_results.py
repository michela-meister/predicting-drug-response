import os

def read_results(data_dir, k_max, s_max, m_max):
    for k in range(1, k_max + 1):
        for r in range(1, k + 1):
            for s in range(1, s_max + 1):
                for m in range(1, m_max + 1):
                    fn = data_dir + '/' + str(r) + '/' + str(k) + '/' + str(s) + '/' + str(m) + '.pkl'
                    if not os.path.isfile(fn):
                        print(str(r) + '/' + str(k) + '/' + str(s) + '/' + str(m))

if __name__ == "__main__":
    data_dir = 'results/2023-07-31/heatmap/REP_GDSC'
    k_max = 20
    s_max = 10
    m_max = 10
    read_results(data_dir, k_max, s_max, m_max)