#!/bin/bash

python3 clean_data.py
python3 add_mids.py
python3 plot_mid_data.py
python3 add_duration.py
python3 preprocessing.py
python3 run_model.py 