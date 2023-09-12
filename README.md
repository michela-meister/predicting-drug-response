This repository accompanies the paper 
Michela Meister, Christopher Tosh, and Wesley Tansey. "Predicting Drug Response via Transfer Learning". (TODO: Add link-to-paper)

*Libraries*
We used:
- Python 3.10.10
- matplotlib 3.4.3
- numpy 1.22.3
- pandas 1.5.3
- pyro-ppl 1.8.4
- pytorch 1.13.1
- scikit-learn 1.2.0
- scipy 1.8.1
- seaborn 0.12.2
- statsmodels 0.13.5
- tqdm 4.65.0
 
Files:
- ```models.py```
- ```expt.py```
- ```pdx_expt.py```
- ```helpers.py```
- fold_info: Holds folds for experiments

Datasets & Cleaning:
- PRISM-REP-GDSC data:
	- Link to paper
	- Which supplemental file to look at
	- Raw data stored here:
	- File for cleaning data here:
	- Cleaned data here:
- PDO-PDX data:
	- Link to paper
	- Which supplemental file to look at
	- Raw data stored here:
	- File for cleaning data here:
	- Cleaned data here:

Results:
- Results from our experiments are stored in ```results``` as files ```experiment1.csv```, ```experiment2.csv```, and ```experiment3.csv```.

Reproducibility:
- To reproduce the plots from the paper, run ```python3 plot_figure1.py```, ```python3 plot_figure2.py```, and ```python3 plot_figure3.py```.
- To run the experiments from the paper, run ```sh ./run_experiment1.sh```, ```sh ./run_experiment2.sh```, and ```sh ./run_experiment3.sh```. Note that these scripts use bsub to submit jobs to an LSF system. 

