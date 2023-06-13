import numpy as np 
import pyro
import random

import pyro.distributions as dist 

import other_helper

pyro.set_rng_seed(0)
print('main file:')
print('numpy: ' + str(np.random.randn()))
a = pyro.sample('a', dist.torch.Normal(loc=0, scale=1))
print('pyro: ' + str(a))
print('random: ' + str(random.choice(range(100))))
print('\n')
pyro.set_rng_seed(0)
other_helper.function_using_randomness()
