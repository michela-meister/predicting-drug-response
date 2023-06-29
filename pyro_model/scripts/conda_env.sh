#!/bin/bash
conda create -n mmenv python=3.9
conda activate mmenv
conda install -c conda-forge pyro-ppl python-graphviz
conda install -c conda-forge seaborn pandas scipy