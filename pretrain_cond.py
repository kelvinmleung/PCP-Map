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
def load_data(data, test_ratio, valid_ratio, random_state, pca_components_x, pca_components_y):
    a_log = np.log(data[326:328, :])
    data[326:328, :] = a_log

    x_data = data[326:, :].T
    y_data = data[:326, :].T

    x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=test_ratio + valid_ratio, random_state=random_state)
    valid_ratio_adjusted = valid_ratio / (test_ratio + valid_ratio)
    x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=valid_ratio_adjusted, random_state=random_state)

    pca_x = PCA(n_components=pca_components_x)
    x_train = pca_x.fit_transform(x_train)
    x_valid = pca_x.transform(x_valid)
    x_test = pca_x.transform(x_test)

    pca_y = PCA(n_components=pca_components_y)
    y_train = pca_y.fit_transform(y_train)
    y_valid = pca_y.transform(y_valid)
    y_test = pca_y.transform(y_test)

    train_mean = x_train.mean(axis=0)
    train_std = x_train.std(axis=0)
    x_train = (x_train - train_mean) / train_std
    x_valid = (x_valid - train_mean) / train_std
    x_test = (x_test - train_mean) / train_std

    y_train_mean = y_train.mean(axis=0)
    y_train_std = y_train.std(axis=0)
    y_train = (y_train - y_train_mean) / y_train_std
    y_valid = (y_valid - y_train_mean) / y_train_std
    y_test = (y_test - y_train_mean) / y_train_std

    return (torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), \
           (torch.tensor(x_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.float32)), \
           (torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

if __name__ == '__main__':

    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__), saving=True)
    logger.info("Starting pretraining at " + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    logger.info(args)

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

    # Load data
    data = np.load(os.path.expanduser(args.data_path), allow_pickle=True)
    print(data.shape)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Multivariate Gaussian as Reference
    input_x_dim = args.pca_components_x
    prior_picnn = distributions.MultivariateNormal(torch.zeros(input_x_dim).to(device),
                                                    torch.eye(input_x_dim).to(device))
    
    for trial in range(args.num_trials):
        reparam = not args.clip
        input_y_dim = args.pca_components_y

        width = np.random.choice(width_list)
        # width_y can be varied
        width_y = width
        batch_size = int(np.random.choice(batch_size_list))
        num_layers = np.random.choice(depth_list)
        lr = np.random.choice(lr_list)

        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(
        data, args.test_ratio, args.valid_ratio, args.random_state, args.pca_components_x, args.pca_components_y)

        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(valid_x, valid_y), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=args.batch_size, shuffle=False)

        picnn = PICNN(input_x_dim, input_y_dim, width, width_y, 1, num_layers, reparam=reparam).to(device)
        pcpmap = PCPMap(prior_picnn, picnn).to(device)
        optimizer = torch.optim.Adam(pcpmap.parameters(), lr=lr)

        # Log the hyperparameters
        logger.info(f"Trial {trial+1}: batch_size={batch_size}, lr={lr}, width={width}, width_y={width_y}, depth={num_layers}")
        params_hist.loc[len(params_hist.index)] = [trial+1, batch_size, lr, width, width_y, num_layers]

        for epoch in range(args.num_epochs):
            pcpmap.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = -pcpmap.loglik_picnn(x_batch, y_batch).mean()
                loss.backward()
                optimizer.step()

                if args.clip:
                    for lw in pcpmap.picnn.Lw:
                        with torch.no_grad():
                            lw.weight.data = pcpmap.picnn.nonneg(lw.weight)

        valLossMeterPICNN = AverageMeter()

        for x_batch, y_batch in valid_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            mean_valid_loss_picnn = -pcpmap.loglik_picnn(x_batch, y_batch).mean()
            valLossMeterPICNN.update(mean_valid_loss_picnn.item(), x_batch.shape[0])

        val_loss_picnn = valLossMeterPICNN.avg
        log_message = f'Trial {trial+1:02d}: Validation Loss = {val_loss_picnn:.3e}'
        logger.info(log_message)
        valid_hist.loc[len(valid_hist.index)] = [trial+1, val_loss_picnn]

    params_hist.to_csv(os.path.join(args.save, 'params_hist.csv'), index=False)
    valid_hist.to_csv(os.path.join(args.save, 'valid_hist.csv'), index=False)