This repository accompanies the paper:        
> Michela Meister, Christopher Tosh, and Wesley Tansey. ["Predicting Drug Response via Transfer Learning."](./predicting-drug-response.pdf) 2023.

### Libraries  
We used
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
 
### Files    
- ```expt.py```: Implements pipeline for simulation studies on cell lines.
- ```pdx_expt.py```: Implements pipeline for simulations studies on PDO-PDX data.
- ```model_helpers.py```: Defines BMT and target-only models.
- ```helpers.py```: Helper functions for reading, writing, and splitting data.
- ```cross_val.py```: Helpers for splitting data during cross-validation.
- ```fold_info```: Directory holding folds for Experiments 1 and 3. 

### Data: Cell Line Datasets     
We studied cell line data from the PRISM, GDSC, and CTD2 datasets provided by the paper below, which introduces the PRISM dataset. The specific dataset we studied is in Supplementary Table 11.
> Corsello, Steven M., et al. "Discovering the anticancer potential of non-oncology drugs by systematic viability profiling." Nature cancer 1.2 (2020): 235-248.
    
The paper accompanying the GDSC dataset is:
> Garnett, Mathew J., et al. "Systematic identification of genomic markers of drug sensitivity in cancer cells." Nature 483.7391 (2012): 570-575.

The paper accomanying the CTD2 dataset is:
> Basu, Amrita, et al. "An interactive resource to identify cancer genetic and lineage dependencies targeted by small molecules." Cell 154.5 (2013): 1151-1161.

*Preprocessing the data:*
- The raw data is ```data/rep-gdsc-ctd2.csv```
- To preprocess the data, run ```python3 code/clean_cell_line_data.py```, which outputs the file ```data/rep-gdsc-ctd2-mean-log.csv```.
    
### Data: PDO-PDX Dataset    
We studied PDO and PDX data provided by the paper below. The specific datasets we studied are in Supplemental Data 14: Drug Sensitivity Data (sheets PDO_Drug_Response, PDO_cetuximab_response, PDX_Drug_Response_TC).
> Schütte, Moritz, et al. "Molecular dissection of colorectal cancer in pre-clinical models identifies biomarkers predicting sensitivity to EGFR inhibitors." Nature communications 8.1 (2017): 14262.

*Preprocessing the data:*
- The raw data is ```data/yaspo_pdo1.csv```, ```data/yaspo_pdo2.csv```, and ```data/yaspo_pdx.csv```.
- To preprocess the data, run ```python3 code/clean_pdo_pdx_data.py```, which outputs the file ```data/yaspo_combined.csv```.

### Results    
- Results from our experiments are ```results/experiment1.csv```, ```results/experiment2.csv```, and ```results/experiment3.csv```.

### Reproducibility    
- To reproduce the plots from the paper, run ```python3 code/plot_figure1.py```, ```python3 code/plot_figure2.py```, and ```python3 code/plot_figure3.py```.
- To run the experiments from the paper, run ```sh scripts/run_experiment1.sh```, ```sh scripts/run_experiment2.sh```, and ```sh scripts/run_experiment3.sh```. Note that these scripts require significant time and computing power to complete. They use bsub to submit jobs to an LSF system.

