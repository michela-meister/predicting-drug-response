def check_args(args, N):
	if len(args) != N:
		print('Error! Expected ' + str(N) + ' arguments, but got ' + str(len(args)))