import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from lib.dataloader import dataloader

# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_sample(y_obs, data, batch_size, test_ratio, valid_ratio, random_state, input_y_dim, input_x_dim, tol, bestParams_picnn):         
    _, _, testData, _ = dataloader(data, batch_size, test_ratio, valid_ratio, random_state)

     # Load Best Models
    model.load_state_dict(bestParams_picnn)
    model = model.to(device)
    
    # Generate samples
    zx = torch.randn(testData.shape[0], input_x_dim).to(device)
    x_generated, _ = model.gx(zx, testData[:, :input_y_dim].to(device), tol=tol)
    x_generated = x_generated.detach().to(device)
   
    # Outputing the average
    x_generated_avg = x_generated.mean(dim=0)

    return x_generated_avg

def plot_for_comparison(data_path, )