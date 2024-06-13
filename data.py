from lib.dataloader import dataloader
import scipy.io
import os
import numpy as np
import pandas as pd
import torch
from torch import distributions
from lib.dataloader import dataloader
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.icnn import PICNN
from src.pcpmap import PCPMap
from src.mmd import mmd
from lib.utils import count_parameters, makedirs, get_logger, AverageMeter
from sklearn.decomposition import PCA
import torch.nn as nn
import matplotlib.pyplot as plt

# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_sample(y_obs, num_sample, input_x_dim, tol, bestParams_picnn_model):         
    model = bestParams_picnn_model.to(device)
    
    # Generate samples
    zx = torch.randn(num_sample, input_x_dim).to(device)
    y_obs = y_obs.to(device)  # Ensure y_obs is on the same device
    x_generated, _ = model.gx(zx, y_obs, tol=tol)
    x_generated = x_generated.detach().to(device)
   
    # Outputing the average without processing
    x_generated_avg = x_generated.mean(dim=0)

    return x_generated, x_generated_avg

def plot_for_comparison(data_path, num_sample, pca_components_x, tol, bestParams_picnn_model, pca_components_y, test_ratio, random_state):         
    data_filename = os.path.basename(data_path)
    data_filename = os.path.splitext(data_filename)[0]
    
    # Construct the paths
    xtruthDir = f"data/x_{data_filename}.npy"
    xisofitMuDir = f"data/x_iso_{data_filename}.npy"
    xisofitGammaDir = f"data/x_iso_gamma_{data_filename}.npy"
    yobsDir = f"data/y_{data_filename}.npy"

    # Load data
    plt.figure()
    wls = np.load('data/wls.npy')
    x_truth = np.load(xtruthDir)
    x_isofit_mu = np.load(xisofitMuDir)
    x_isofit_gamma = np.load(xisofitGammaDir)
    y_obs = np.load(yobsDir)
    data = np.load(data_path, allow_pickle=True)
    n = len(wls)

    # Preprocessing
    a_log = np.log(data[326:328, :])
    data[326:328, :] = a_log
    x_data = data[326:, :].T
    y_data = data[:326, :].T

    pca_x = PCA(n_components=pca_components_x)
    x_data_pca = pca_x.fit_transform(x_data)
    pca_y = PCA(n_components=pca_components_y)
    y_data_pca = pca_y.fit_transform(y_data)
    data_pca = np.concatenate((x_data_pca, y_data_pca), axis=1)
    train, valid = train_test_split(data_pca, test_size=test_ratio, random_state=random_state)

    train_mean = np.mean(train, axis=0, keepdims=True)
    train_std = np.std(train, axis=0, keepdims=True)

    train_mean = torch.tensor(train_mean, dtype=torch.float32).to(device)
    train_std = torch.tensor(train_std, dtype=torch.float32).to(device)

    # Getting sample
    if y_obs.ndim == 1:
        y_obs = y_obs.reshape(-1, 1)
    y_reduced = pca_y.transform(y_obs.T)
    y_reduced = torch.tensor(y_reduced, dtype=torch.float32).to(device)
    y_normalised = (y_reduced - train_mean[:, :pca_components_y]) / train_std[:, :pca_components_y]
    y_normalised = torch.tensor(y_normalised, dtype=torch.float32).to(device)
    x_generated, _ = generate_sample(y_normalised, num_sample, pca_components_x, tol, bestParams_picnn_model)
    
    # Process x
    if x_generated.ndim == 1:
        x_generated = x_generated.reshape(-1, 1)

    x_generated = (x_generated * train_std[:, pca_components_y:]) + train_mean[:, pca_components_y:]
    x_generated = pca_x.inverse_transform(x_generated)

    a_orig = np.exp(np.clip(x_generated[:, :2], a_min=-700, a_max=700))
    x_generated[:, :2] = a_orig

    X_star_refl = x_generated[:, 2:]
    mu_pos = np.mean(X_star_refl, 0)
    gamma_pos = np.cov(X_star_refl.T)

    plt.figure()
    plt.plot(wls, x_truth, 'r', alpha=1, linewidth=3, label="Truth")
    plt.plot(wls, x_isofit_mu, 'b', alpha=0.7, label="Pos MAP - Isofit")
    plt.plot(wls, mu_pos, 'g', alpha=0.7, label="Posterior - Transport")
    plt.axvspan(1300, 1450, alpha=0.8, color='black')
    plt.axvspan(1780, 2050, alpha=0.8, color='black')
    plt.ylim(bottom=0)
    plt.xlabel("Wavelength")
    plt.ylabel("Reflectance")
    plt.title("Posterior mean")
    plt.legend()
    plt.savefig(f'plots/refl_pos_mean_{data_filename}.png', dpi=300)

    plt.figure()
    plt.plot(wls, np.diag(x_isofit_gamma), 'b', alpha=0.7, label="Pos MAP - Isofit")
    plt.plot(wls, np.diag(gamma_pos), 'g', alpha=0.7, label="Posterior - Transport")
    plt.axvspan(1300, 1450, alpha=0.8, color='black')
    plt.axvspan(1780, 2050, alpha=0.8, color='black')
    plt.ylim(bottom=0, top=0.0001)
    plt.xlabel("Wavelength")
    plt.ylabel("Variance")
    plt.title("Posterior marginal variance")
    plt.legend()
    plt.savefig(f'plots/refl_pos_var_{data_filename}.png', dpi=300)

    data = np.load(data_path, allow_pickle=True).T
    plt.figure()
    plt.scatter(data[:, 326], data[:, 327], alpha=0.5, label='Prior')
    plt.scatter(x_generated[:, 0], x_generated[:, 1], alpha=0.5, label='Transport')
    plt.xlabel('AOD')
    plt.ylabel('H2O')
    plt.legend()
    plt.title('Posterior samples - atmosphere')
    plt.savefig(f'plots/atm_pos_samp_{data_filename}.png', dpi=300)

    return x_generated

checkpoint_path = 'experiments/cond/177/177_2024_06_11_21_52_36_32_0.01_3_256_checkpt.pth'
checkpoint = torch.load(checkpoint_path)

input_x_dim = 40
input_y_dim = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prior_picnn = distributions.MultivariateNormal(torch.zeros(input_x_dim).to(device), torch.eye(input_x_dim).to(device))

# Build PCP-Map
picnn = PICNN(input_x_dim, input_y_dim, 256, 256, 1, 3, reparam=False)
pcpmap = PCPMap(prior_picnn, picnn).to(device)

pcpmap.load_state_dict(checkpoint['state_dict_picnn'])

x_generated = plot_for_comparison('ens/177.p', 100, 40, 1e-06, pcpmap, 40, 0.1, 42)
print(x_generated)
