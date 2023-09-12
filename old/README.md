Welcome to the drug-response transfer learning project.

Datasets used:
GDSC, REP, CTD2

GDSC-REP-CTD2 DATA
Link to original paper, datasets:
Uncleaned data: data/data/rep-gdsc-ctd2.csv

Cleaning the data: 
cd to pyro_model
sh ./new_scripts clean_dataset.sh --> outputs file data/rep-gdsc-ctd2-mean-log.csv

Main experiment:
code/expt.py -- runs entire pipeline for a given pair of datasets and a fold / random seed

Model file:
code/model_helpers.py -- Edit this file to change the transfer model. 

Ways to test your code:
new_scripts/test_transfer.py -- runs code/expt.py for a given set of parameters. If you want to run faster (ie to debug locally), set n_steps to 5 instead of 1000. This will run the entire choose_k pipeline (which can also take a long time)

Data folds:
folds_info/fold_list_10.pkl -- List of folds. Each fold is a list of sample_id's. 


Other helper files:
code/helpers.py -- general helpers for reading and writing data, and for splitting the dataset into train and test
code/cross_val.py -- helpers for the cross-validation process used to choose parameter k

Running scripts on the cluster:
- conda environment used on cluster is saved in meister_env_2023-08-15.yml
- Main tip: parallelize as much as possible
- See scripts like run_2c.sh for examples of how to run many jobs on the cluster in parallel over different dataset pairs, methods, etc
