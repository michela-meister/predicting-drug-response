# model initialization parameters
#PARAMS = {'a_sigma_init': 5, 'g_alpha_init': 10, 'g_beta_init': 2, 'alpha_init': 2, 'beta_init': 1}
GAMMA_MEAN = 1.5
GAMMA_VARIANCE = 100.00
#PARAMS = {'alpha': (GAMMA_MEAN**2) / GAMMA_VARIANCE, 'beta': GAMMA_MEAN / GAMMA_VARIANCE}
PARAMS = {'alpha': 10, 'beta': 2}
# lower and upper bounds for coverage
LO = .05
HI = .95
FRACTION_TRAIN = .75
# mcmc params
N_MCMC = 500
N_WARMUP = 500