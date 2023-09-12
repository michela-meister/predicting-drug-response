import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import sys

NUM_ARGS = 4
n = len(sys.argv)
if n != NUM_ARGS:
    print("Error: " + str(NUM_ARGS) + " arguments needed; only " + str(n) + " arguments given.")
read_fn = sys.argv[1].split("=")[1]
write_dir = sys.argv[2].split("=")[1]
test_fn = sys.argv[3].split("=")[1]

# read in mcmc samples
with open(read_fn, 'rb') as handle:
    mcmc_samples = pickle.load(handle)


# read in test data -- do I want test data as (sample, drug, obs) pairs? (instead of obs_list?)


# Idea for one round:
# sample all parameters from mcmc samples
# for each (sample, drug) combo in the test set: 
##### use the params to generate a set S of synthetic samples
##### sort the samples in S
##### compute the mean of S
##### compute the 5th and 95th percentiles of S
##### for each test observation with the given (sample, drug) pair:
########### Does the obs fall between the 5th and 95th percentiles? (bool)
########### What is the distance to the mean? (float)
########### Store test obs, mean if plotting
# Compute the R^2 for (obs, mean) over all (sample, drug) pairs
# Compute the fraction of samples within the 5th, 95th percentiles
# Store the R^2 and the fractions in (5, 95)

# After all rounds, can plot the R^2 values and the fraction within 5th, 95th percentiles
