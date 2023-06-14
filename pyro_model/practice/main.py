import numpy as np 
import pyro
import random
import sys

import pyro.distributions as dist 

import test_import_globals as const
import other_helper

# pyro.set_rng_seed(0)
# print('main file:')
# print('numpy: ' + str(np.random.randn()))
# a = pyro.sample('a', dist.torch.Normal(loc=0, scale=1))
# print('pyro: ' + str(a))
# print('random: ' + str(random.choice(range(100))))
# print('\n')
# pyro.set_rng_seed(0)
# other_helper.function_using_randomness()

def check_args(args, N):
	if len(args) != N:
		print('Error! Expected ' + str(N) + ' arguments, but got ' + str(len(args)))

def get_args(args):
	arg = int(args[1].split("=")[1])
	print('The argument passed in is: ' + str(arg))

def main():
	check_args(sys.argv, 2)
	print(const.FIRST_WORD + '\n' + const.SECOND_WORD)
	get_args(sys.argv)

if __name__ == "__main__":
    main()