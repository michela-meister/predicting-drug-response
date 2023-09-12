import numpy as np 
import pandas as pd 
import pyro
import torch

import global_constants as const 
import model_helpers as modeling

n_samp = 100
n_drug = 80
s_idx1 = [1, 2, 3]
d_idx1 = [1, 2, 3]
s_idx2 = [1, 2, 3]
d_idx2 = [1, 2, 3]
params = const.PARAMS 
obs_1 = torch.Tensor([10, 20, 30])
obs_2 = torch.Tensor([100, 200, 300])

pyro.render_model(modeling.transfer_model, model_args=(n_samp, n_drug, s_idx1, d_idx1, s_idx2, d_idx2, params, obs_1, len(obs_1), obs_2, len(obs_2)), render_distributions=True,
	filename='model_diagram.png')