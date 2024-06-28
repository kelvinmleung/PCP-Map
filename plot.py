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
import argparse

parser = argparse.ArgumentParser('PCP-Map')
parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the checkpoint file')
parser.add_argument('--extra_plots', type=str, default=False, help='Path to the checkpoint file')

args = parser.parse_args()

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

def plot_for_comparison(data, num_sample, pca_components_s, tol, bestParams_picnn_model, pca_components_y, test_ratio, random_state, data_type, more_plot=False):         
    data_filename = data 
    
    if data_type == 'real':
        data_path = f"ens/{data_filename}.p"
        yobsDir = f"data/y_{data_filename}.npy"
        mcmcDir = f"mcmc/real/mcmc_{data_filename}.npy"
    else:
        data_path = f"ensembles_a=[0.2,1.5]/ens_{data_filename}.npy"
        yobsDir = f"ensembles_a=[0.2,1.5]/yobs_sim_{data_filename}.npy"
        mcmcDir = f"mcmc/synthetic/mcmc_simobs_{data_filename}.npy"

    # Construct the paths
    xtruthDir = f"data/x_{data_filename}.npy"
    xisofitMuDir = f"data/x_iso_{data_filename}.npy"
    xisofitGammaDir = f"data/x_iso_gamma_{data_filename}.npy"

    # Load data
    plt.figure()
    wls = np.load('data/wls.npy')
    x_truth = np.load(xtruthDir)
    x_isofit_mu = np.load(xisofitMuDir)
    x_isofit_gamma = np.load(xisofitGammaDir)
    y_obs = np.load(yobsDir)
    data = np.load(data_path, allow_pickle=True)
    n = len(wls)

    # determine year
    if data_filename in ["177", "306", "dark", "mars"]:
        year = '2014'
    else:
        year = '2017'

    # load mcmc
    if os.path.exists(mcmcDir):
        mcmcChain = np.load(mcmcDir)
        mcmcChain = mcmcChain[:, ::20]
        bands = np.load("data/wl_ind_" + year + ".npy")
        mcmcmean = np.mean(mcmcChain[bands,:],1)
        mcmccov = np.cov(mcmcChain[bands,:])
    else:
        mcmcChain = None
        mcmcmean = None
        mcmccov = None

    # Preprocessing
    a_log = np.log(data[326:328, :]).T
    
    s_data = data[328:, :].T
    y_data = data[:326, :].T

    pca_s = PCA(n_components=pca_components_s)
    s_data_pca = pca_s.fit_transform(s_data)
    pca_y = PCA(n_components=pca_components_y)
    y_data_pca = pca_y.fit_transform(y_data)

    if data_type == 'real':
        data_pca = np.concatenate((y_data_pca, a_log, s_data_pca), axis=1)
    else:
        data_pca = np.concatenate((y_data_pca, s_data_pca), axis=1)

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

    if data_type == 'real':
        input_dim = pca_components_s + 2
    else:
        input_dim = pca_components_s 
    x_generated, _ = generate_sample(y_normalised, num_sample, input_dim, tol, bestParams_picnn_model)
    
    # Process x
    if x_generated.ndim == 1:
        x_generated = x_generated.reshape(-1, 1)

    x_generated = (x_generated * train_std[:, pca_components_y:]) + train_mean[:, pca_components_y:]

    if data_type == 'real':
        s_generated = x_generated[:, 2:]
    else:
        s_generated = x_generated
    s_generated = pca_s.inverse_transform(s_generated)

    if data_type == 'real':
        a_orig = np.exp(x_generated[:, :2])
    else:
        a_orig = data[326:328, :].T

    X_star = np.concatenate((a_orig, s_generated), axis=1)

    X_star_refl = X_star[:,2:]
    mu_pos = np.mean(X_star_refl, 0)
    gamma_pos = np.cov(X_star_refl.T)

    data_filename = os.path.basename(data_path)
    data_filename = os.path.splitext(data_filename)[0]
    plt.figure()
    plt.plot(wls, x_truth, 'r', alpha=1, linewidth=3, label="Truth")
    if data_type == 'real':
        plt.plot(wls, x_isofit_mu, 'b', alpha=0.7, label="Pos MAP - Isofit")
    plt.plot(wls, mu_pos, 'g', alpha=0.7, label="Posterior - Transport")
    if mcmcmean is not None:
        plt.plot(wls, mcmcmean, 'k', alpha=0.7, label="Posterior - MCMC")
    plt.axvspan(1300, 1450, alpha=0.8, color='black')
    plt.axvspan(1780, 2050, alpha=0.8, color='black')
    plt.ylim(bottom=0)
    plt.xlabel("Wavelength")
    plt.ylabel("Reflectance")
    plt.title("Posterior mean")
    plt.legend()
    if data_type == 'real':
        plt.savefig(f'plots/refl_pos_mean_{data_filename}.png', dpi=300)
    else:
        plt.savefig(f'plots/refl_pos_mean_{data_filename}_synthetic.png', dpi=300)

    plt.figure()
    if data_type == 'real':
        plt.plot(wls, np.diag(x_isofit_gamma), 'b', alpha=0.7, label="Pos MAP - Isofit")
    plt.plot(wls, np.diag(gamma_pos), 'g', alpha=0.7, label="Posterior - Transport")
    if mcmccov is not None:
        plt.plot(wls, np.diag(mcmccov), 'k', alpha=0.7, label="Posterior - MCMC")
    plt.axvspan(1300, 1450, alpha=0.8, color='black')
    plt.axvspan(1780, 2050, alpha=0.8, color='black')
    plt.ylim(bottom=0)
    plt.xlabel("Wavelength")
    plt.ylabel("Variance")
    plt.title("Posterior marginal variance")
    plt.legend()
    if data_type == 'real':
        plt.savefig(f'plots/refl_pos_var_{data_filename}.png', dpi=300)
    else:
        plt.savefig(f'plots/refl_pos_var_{data_filename}_synthetic.png', dpi=300)

    if data_type == 'real':
        data = np.load(data_path, allow_pickle=True).T
        plt.figure()
        plt.scatter(data[:, 326], data[:, 327], alpha=0.5, label='Prior')
        plt.scatter(X_star[:, 0], X_star[:, 1], alpha=0.5, label='Transport')
        if mcmcChain is not None:
            plt.scatter(mcmcChain[-2, :], mcmcChain[-1, :], alpha=0.5, label='MCMC')
        plt.xlabel('AOD')
        plt.ylabel('H2O')
        plt.ylim((0,4))
        plt.legend()
        plt.title('Posterior samples - atmosphere')
        plt.savefig(f'plots/atm_pos_samp_{data_filename}.png', dpi=300)

    if mcmcChain is not None and more_plot:
        relerr_isofit = abs(x_isofit_mu - mcmcmean) / abs(mcmcmean)
        relerr_transport = abs(mu_pos - mcmcmean) / abs(mcmcmean)
        weighterr_isofit = abs(x_isofit_mu - mcmcmean) / np.sqrt(np.diag(x_isofit_gamma))
        weighterr_transport = abs(mu_pos - mcmcmean) / np.sqrt(np.diag(gamma_pos))

        plt.figure()
        plt.plot(wls, np.sqrt(np.diag(x_isofit_gamma)), 'b', alpha=0.7, label="Pos MAP - Isofit")
        plt.plot(wls, np.sqrt(np.diag(mcmccov)), 'k', alpha=0.7, label="Posterior - MCMC")
        plt.plot(wls, np.sqrt(np.diag(gamma_pos)), 'g', alpha=0.7, label="Posterior - Transport")
        plt.axvspan(1300, 1450, alpha=0.8, color='black')
        plt.axvspan(1780, 2050, alpha=0.8, color='black')
        plt.ylim(bottom=0, top=0.006)
        plt.xlabel("Wavelength")
        plt.ylabel("Standard Deviation")
        plt.title("Posterior marginal stddev")
        plt.legend()
        if data_type == 'real':
            plt.savefig(f'plots/refl_pos_std_{data_filename}.png', dpi=300)
        else:
            plt.savefig(f'plots/refl_pos_std_{data_filename}_synthetic.png', dpi=300)

        plt.figure()
        plt.plot(wls, abs(x_isofit_mu - mcmcmean), 'b', alpha=0.7, label="Isofit")
        plt.plot(wls, abs(mu_pos - mcmcmean),'g', alpha=0.7, label="Transport")
        plt.axvspan(1300, 1450, alpha=0.8, color='black')
        plt.axvspan(1780, 2050, alpha=0.8, color='black')
        plt.ylim(bottom=0, top=0.006)
        plt.xlabel("Wavelength")
        plt.ylabel("Absolute difference from MCMC")
        plt.title("Posterior mean difference")
        plt.legend()
        if data_type == 'real':
            plt.savefig(f'plots/pos_mean_diff_{data_filename}.png', dpi=300)
        else:
            plt.savefig(f'plots/pos_mean_diff_{data_filename}_synthetic.png', dpi=300)

        plt.figure()
        plt.plot(wls, relerr_isofit, 'b', alpha=0.7, label="Isofit")
        plt.plot(wls, relerr_transport,'g', alpha=0.7, label="Transport")
        plt.axvspan(1300, 1450, alpha=0.8, color='black')
        plt.axvspan(1780, 2050, alpha=0.8, color='black')
        plt.xlabel("Wavelength")
        plt.ylabel("Relative error")
        plt.title("Relative error in posterior mean")
        plt.legend()
        if data_type == 'real':
            plt.savefig(f'plots/pos_mean_relative_err_{data_filename}.png', dpi=300)
        else:
            plt.savefig(f'plots/pos_mean_relative_err_{data_filename}_synthetic.png', dpi=300)

        plt.figure()
        plt.plot(wls, weighterr_isofit, 'b', alpha=0.7, label="Isofit")
        plt.plot(wls, weighterr_transport,'g', alpha=0.7, label="Transport")
        plt.axvspan(1300, 1450, alpha=0.8, color='black')
        plt.axvspan(1780, 2050, alpha=0.8, color='black')
        plt.ylim(bottom=0)
        plt.xlabel("Wavelength")
        plt.ylabel("Error in mean weighted by precision")
        plt.title("Posterior weighted error")
        plt.legend()
        if data_type == 'real':
            plt.savefig(f'plots/pos_weighted_err_{data_filename}.png', dpi=300)
        else:
            plt.savefig(f'plots/pos_weighted_err_{data_filename}_synthetic.png', dpi=300)

        data = np.load(data_path, allow_pickle=True).T
        plt.figure()
        b1, b2 = [50, 100]
        plt.scatter(data[:,328+b1], data[:,328+b2], alpha=0.1,label='Prior')
        plt.scatter(mcmcChain[b1,:], mcmcChain[b2,:], alpha=0.1,label='MCMC')
        plt.scatter(X_star[:,b1-2], X_star[:,b2-2], alpha=0.1,label='Transport')
        plt.xlabel('AOD')
        plt.ylabel('H2O')
        plt.legend()
        plt.title('Posterior samples')
        if data_type == 'real':
            plt.savefig(f'plots/pos_samples_{data_filename}.png', dpi=300)
        else:
            plt.savefig(f'plots/pos_samples_{data_filename}_synthetic.png', dpi=300)

    return X_star

if __name__ == "__main__":
    # checkpoint_path = 'experiments/cond/ens_mars/ens_mars_2024_06_24_06_20_43_64_0.01_6_256_checkpt.pth'
    checkpoint_path = args.checkpoint_path

    checkpoint = torch.load(checkpoint_path)

    saved_args = checkpoint['args']

    input_s_dim = saved_args.input_s_dim
    input_y_dim = saved_args.input_y_dim
    feature_dim = saved_args.feature_dim
    feature_y_dim = saved_args.feature_y_dim
    num_layers = saved_args.num_layers_pi
    reparam = not saved_args.clip 
    tol = saved_args.tol
    pca_components_s = saved_args.pca_components_s
    pca_components_y = saved_args.pca_components_y
    test_ratio = saved_args.test_ratio
    random_state = saved_args.random_state

    data_type = saved_args.data_type
    data = saved_args.data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_type == 'real':
        input_dim = pca_components_s + 2
    else:
        input_dim = pca_components_s
    prior_picnn = distributions.MultivariateNormal(torch.zeros(input_dim).to(device), torch.eye(input_dim).to(device))

    # Build PCP-Map
    picnn = PICNN(input_dim, pca_components_y, feature_dim, feature_y_dim, 1, num_layers, reparam=reparam)
    pcpmap = PCPMap(prior_picnn, picnn).to(device)

    pcpmap.load_state_dict(checkpoint['state_dict_picnn'])

    x_generated = plot_for_comparison(data, 10000, pca_components_s, tol, pcpmap, pca_components_y, test_ratio, random_state, data_type, args.extra_plots) 

    print(x_generated)
