import argparse
import os
import numpy as np
import datetime
import pandas as pd
import torch
import scipy.io
import pickle
from torch import distributions
# from lib.dataloader import dataloader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from src.icnn import PICNN
from src.pcpmap import PCPMap
from lib.utils import makedirs, get_logger, AverageMeter
from sklearn.decomposition import PCA

"""
argument parser for hyper parameters and model handling
"""
# Define argument parser
parser = argparse.ArgumentParser('PCP-Map Pretraining')
parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset pickle file")
parser.add_argument('--input_x_dim', type=int, default=328, help="Input data convex dimension")
parser.add_argument('--input_y_dim', type=int, default=326, help="Input data non-convex dimension")
parser.add_argument('--pca_components_x', type=int, default=40, help="Number of PCA components for x data")
parser.add_argument('--pca_components_y', type=int, default=40, help="Number of PCA components for y data")
parser.add_argument('--num_epochs', type=int, default=1, help="Number of pre-training epochs")
parser.add_argument('--test_ratio', type=float, default=0.10, help="Test set ratio")
parser.add_argument('--valid_ratio', type=float, default=0.10, help="Validation set ratio")
parser.add_argument('--random_state', type=int, default=42, help="Random state for splitting dataset")
parser.add_argument('--save', type=str, default='~/code/PCP-Map/experiments/tabcond', help="Directory to save results")
parser.add_argument('--clip', type=bool, default=True, help="Whether to clip the weights or not")
parser.add_argument('--tol', type=float, default=1e-12, help="LBFGS tolerance")
parser.add_argument('--num_trials', type=int, default=5, help="Number of pilot runs")
args, unknown = parser.parse_known_args()

sStartTime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# logger
makedirs(args.save)
logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__), saving=True)
logger.info("start time: " + sStartTime)
logger.info(args)

# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading function
def load_data(data, test_ratio, valid_ratio, batch_size, random_state, pca_components_x, pca_components_y):
    # enforce positivity
    # below the row_numbers are specified according to remote sensing context
    a_log = np.log(data[326:328, :])
    data[326:328, :] = a_log

    x_data = data[326:, :].T
    y_data = data[:326, :].T

    pca_x = PCA(n_components=pca_components_x)
    x_data = pca_x.fit_transform(x_data)
    pca_y = PCA(n_components=pca_components_y)
    y_data = pca_y.fit_transform(y_data)

    data = np.concatenate((x_data, y_data), axis=1)

    # split data and convert to tensor
    train, valid = train_test_split(
        data, test_size=test_ratio,
        random_state=random_state
    )
    train_sz = train.shape[0]

    train_mean = np.mean(train, axis=0, keepdims=True)
    train_std = np.std(train, axis=0, keepdims=True)
    train_data = (train - train_mean) / train_std
    valid_data = (valid - train_mean) / train_stdí

    # convert to tensor
    train_data = torch.tensor(train, dtype=torch.float32)
    valid_data = torch.tensor(valid, dtype=torch.float32)

    # load train data
    trn_loader = DataLoader(
        train_data,
        batch_size=batch_size, shuffle=True
    )
    vld_loader = DataLoader(
        valid_data,
        batch_size=batch_size, shuffle=True
    )

    return trn_loader, vld_loader, train_sz

if __name__ == '__main__':
    columns_params = ["batchsz", "lr", "width", "width_y", "depth"]
    columns_valid = ["picnn_nll"]
    params_hist = pd.DataFrame(columns=columns_params)
    valid_hist = pd.DataFrame(columns=columns_valid)

    log_msg = ('{:5s}  {:9s}'.format('trial', 'val_loss'))
    logger.info(log_msg)

    # sample space for hyperparameters ## TO BE CHANGED
    width_list = np.array([32, 64, 128, 256, 512])
    depth_list = np.array([2, 3, 4, 5, 6])
    batch_size_list =  np.array([32, 64, 128])
    lr_list = np.array([0.01, 0.001, 0.0001])

    for trial in range(args.num_trials):
        # randomly initialize hyperparameters
        reparam = not args.clip
        width = np.random.choice(width_list)
        # width_y can be varied
        width_y = width
        batch_size = int(np.random.choice(batch_size_list))
        num_layers = np.random.choice(depth_list)
        lr = np.random.choice(lr_list)

        # load data
        data_path = os.path.expanduser(args.data_path)
        data = np.load(data_path, allow_pickle=True)
        train_loader, valid_loader, _ = load_data(
            data, args.test_ratio, args.valid_ratio, batch_size, args.random_state, args.pca_components_x, args.pca_components_y)

        # Multivariate Gaussian as Reference
        input_x_dim = args.pca_components_x
        prior_picnn = distributions.MultivariateNormal(torch.zeros(input_x_dim).to(device),
                                                        torch.eye(input_x_dim).to(device))

        # build PCP-Map
        input_y_dim = args.pca_components_y
        picnn = PICNN(input_x_dim, input_y_dim, width, width_y, 1, num_layers, reparam=reparam).to(device)
        pcpmap = PCPMap(prior_picnn, picnn).to(device)
        optimizer = torch.optim.Adam(pcpmap.parameters(), lr=lr)

        params_hist.loc[len(params_hist.index)] = [batch_size, lr, width, width_y, num_layers]

        for epoch in range(args.num_epochs):
            for sample in train_loader:
                y = sample[:,:input_y_dim].requires_grad_(True).to(device)
                x = sample[:,input_y_dim:].requires_grad_(True).to(device)

                optimizer.zero_grad()
                loss = -pcpmap.loglik_picnn(x, y).mean()
                loss.backward()
                optimizer.step()

                # non-negative constraint
                if args.clip:
                    for lw in pcpmap.picnn.Lw:
                        with torch.no_grad():
                            lw.weight.data = pcpmap.picnn.nonneg(lw.weight)

        valLossMeterPICNN = AverageMeter()

        for valid_sample in valid_loader:
            y_valid = valid_sample[:,:input_y_dim].requires_grad_(True).to(device)
            x_valid = valid_sample[:,input_y_dim:].requires_grad_(True).to(device)
            mean_valid_loss_picnn = -pcpmap.loglik_picnn(x_valid, y_valid).mean()
            valLossMeterPICNN.update(mean_valid_loss_picnn.item(), x.shape[0])

        val_loss_picnn = valLossMeterPICNN.avg

        log_message = '{:05d}  {:9.3e}'.format(trial+1, val_loss_picnn)
        logger.info(log_message)
        valid_hist.loc[len(valid_hist.index)] = [val_loss_picnn]
    
    data_filename = os.path.basename(args.data_path)
    data_filename = os.path.splitext(data_filename)[0]
    params_hist.to_csv(os.path.join(args.save, f'{data_filename}_params_hist.csv'))
    valid_hist.to_csv(os.path.join(args.save, f'{data_filename}_valid_hist.csv'))