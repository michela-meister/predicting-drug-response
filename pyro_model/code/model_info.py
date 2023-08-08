import numpy as np 

import helpers

# K-values for each method x dataset pair. In our models r = k

transfer_fn = 'model_info/kvals_transfer.pkl'
transfer = {'log_CTD2_GDSC': 30, 'log_CTD2_REP': 28, 'log_GDSC_CTD2': 31, 'log_GDSC_REP': 35, 'log_REP_GDSC': 29, 'log_REP_CTD2': 29}
helpers.write_pickle(transfer, transfer_fn)

target_only = {'log_CTD2': 37, 'log_GDSC': 41, 'log_REP': 42}
target_only_fn = 'model_info/kvals_target_only.pkl'
helpers.write_pickle(target_only, target_only_fn)