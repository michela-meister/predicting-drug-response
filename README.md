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
- expt.py: Implements pipeline for simulation studies on cell lines.
- pdx_expt.py: Implements pipeline for simulations studies on POD-PDX data.
- model_helpers.py: Defines BMT and target-only models.
- helpers.py: Helper functions for reading, writing, and splitting data.
- fold_info: Holds files with folds for Experiments 1 and 3. 

Datasets & Cleaning:
- PRISM-GDSC-CTD2:
	- Paper link: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7328899/
	- Excel link: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7328899/bin/NIHMS1589633-supplement-Supplementary_Tables.xlsx (Supplementary Table 11)
	- The raw data is ```data/rep-gdsc-ctd2.csv```
	- To preprocess the raw data, run ```python3 code/clean_cell_line_data.py```, which outputs the file ```data/rep-gdsc-ctd2-mean-log.csv```.
- PDO-PDX data:
	- Paper link: https://www.nature.com/articles/ncomms14262 See Supplemental Data 14: Drug Sensitivity Data
	- PDO data from sheets: PDO_Drug_Response, PDO_cetuximab_response
	- PDX data from sheet: PDX_Drug_Response_TC
	- The raw data is ```data/yaspo_pdo1.csv```, ```data/yaspo_pdo2.csv```, and ```data/yaspo_pdx.csv```.
	- To preprocess the data, run ```python3 code/clean_pdo_pdx_data.py```, which outputs the file ```data/yaspo_combined.csv```.

Results:
- Results from our experiments are stored in ```results``` as files ```experiment1.csv```, ```experiment2.csv```, and ```experiment3.csv```.

Reproducibility:
- To reproduce the plots from the paper, run ```python3 code/plot_figure1.py```, ```python3 code/plot_figure2.py```, and ```python3 code/plot_figure3.py```.
- To run the experiments from the paper, run ```sh ./run_experiment1.sh```, ```sh ./run_experiment2.sh```, and ```sh ./run_experiment3.sh```. Note that these scripts require significant time and computing power to complete. They use bsub to submit jobs to an LSF system.

