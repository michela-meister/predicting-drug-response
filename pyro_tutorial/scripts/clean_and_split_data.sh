#!/bin/bash
base_dir=code
python3 $base_dir/clean_data.py
python3 $base_dir/add_mids.py
python3 $base_dir/plot_mid_data.py
python3 $base_dir/add_duration.py
python3 $base_dir/preprocessing.py