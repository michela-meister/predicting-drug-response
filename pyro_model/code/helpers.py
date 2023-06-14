import pickle

def check_args(args, N):
	if len(args) != N:
		print('Error! Expected ' + str(N) + ' arguments, but got ' + str(len(args)))

def write_to_pickle(obj, fn):
    with open(fn, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(fn):
    with open(fn, 'rb') as handle:
        obj = pickle.load(handle)
    return obj