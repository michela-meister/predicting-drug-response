import numpy as np 
import pyro
import random

import pyro.distributions as dist 


def function_using_randomness():
	print('function from other file: ')
	print('numpy: ' + str(np.random.randn()))
	a = pyro.sample('a', dist.torch.Normal(loc=0, scale=1))
	print('pyro: ' + str(a))
	print('random: ' + str(random.choice(range(100))))