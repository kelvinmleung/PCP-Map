import argparse
import os
import time
import datetime
import scipy.io
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

"""
argument parser for hyper parameters and model handling
"""

parser = argparse.ArgumentParser('PCP-Map')
# parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset pickle file")
parser.add_argument('--data', type=str, choices=['177', '306', 'mars', 'dark', 'beckmen'], required=True, help="Identifier for the dataset (e.g., '177')")
parser.add_argument('--data_type', type=str, required=True, help="Type of the dataset ('real' or 'synthetic')")

parser.add_argument('--input_x_dim',    type=int, default=328, help="input data convex dimension")
parser.add_argument('--input_s_dim', type=int, default=326, help="Input data convex dimension")
parser.add_argument('--input_y_dim',    type=int, default=326, help="input data non-convex dimension")
parser.add_argument('--feature_dim',    type=int, default=128, help="intermediate layer feature dimension")
parser.add_argument('--feature_y_dim',  type=int, default=128, help="intermediate layer context dimension")
parser.add_argument('--pca_components_s', type=int, default=40, help="Number of PCA components for s data")
parser.add_argument('--pca_components_y', type=int, default=40, help="Number of PCA components for y data")

parser.add_argument('--num_layers_pi',  type=int, default=2, help="depth of PICNN network")

parser.add_argument('--clip',           type=bool, default=True, help="whether clipping the weights or not")
parser.add_argument('--tol',            type=float, default=1e-6, help="LBFGS tolerance")

parser.add_argument('--batch_size',     type=int, default=256, help="number of samples per batch")
parser.add_argument('--num_epochs',     type=int, default=10, help="number of training steps")
parser.add_argument('--print_freq',     type=int, default=1, help="how often to print results to log")
parser.add_argument('--valid_freq',     type=int, default=50, help="how often to run model on validation set")
parser.add_argument('--early_stopping', type=int, default=20, help="early stopping of training based on validation")
parser.add_argument('--lr',             type=float, default=0.005, help="optimizer learning rate")
parser.add_argument("--lr_drop",        type=float, default=2.0, help="how much to decrease lr (divide by)")

parser.add_argument('--test_ratio',     type=float, default=0.10, help="test set ratio")
parser.add_argument('--valid_ratio',    type=float, default=0.10, help="validation set ratio")
parser.add_argument('--random_state',   type=int, default=42, help="random state for splitting dataset")

parser.add_argument('--save_test',      type=int, default=1, help="if 1 then saves test numerics 0 if not")
parser.add_argument('--save',           type=str, default='experiments/cond', help="define the save directory")

args = parser.parse_args()

sStartTime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# logger
makedirs(args.save)
logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__), saving=True)
logger.info("start time: " + sStartTime)
logger.info(args)

# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# decrease the learning rate based on validation
ndecs_picnn = 0
n_vals_wo_improve_picnn = 0
def update_lr_picnn(optimizer, n_vals_without_improvement):
    global ndecs_picnn
    if ndecs_picnn == 0 and n_vals_without_improvement > args.early_stopping:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop
        ndecs_picnn = 1
    elif ndecs_picnn == 1 and n_vals_without_improvement > args.early_stopping:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop**2
        ndecs_picnn = 2
    else:
        ndecs_picnn += 1
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop**ndecs_picnn


def load_data(data, test_ratio, valid_ratio, batch_size, random_state, pca_components_s, pca_components_y, data_type):
    a_log = np.log(data[326:328, :]).T

    s_data = data[328:, :].T
    y_data = data[:326, :].T

    pca_s = PCA(n_components=pca_components_s)
    s_data = pca_s.fit_transform(s_data)
    pca_y = PCA(n_components=pca_components_y)
    y_data = pca_y.fit_transform(y_data)

    if data_type == 'real':
        data = np.concatenate((y_data, a_log, s_data), axis=1)
    else:
        data = np.concatenate((y_data, s_data), axis=1)

    # split data and convert to tensor
    train, valid = train_test_split(
        data, test_size=test_ratio,
        random_state=random_state
    )
    train_sz = train.shape[0]

    train_mean = np.mean(train, axis=0, keepdims=True)
    train_std = np.std(train, axis=0, keepdims=True)
    train_data = (train - train_mean) / train_std
    valid_data = (valid - train_mean) / train_std

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


def evaluate_model(model, data, batch_size, test_ratio, valid_ratio, random_state, input_y_dim, input_x_dim, tol,
                   bestParams_picnn):

    _, _, testData, _ = dataloader(data, batch_size, test_ratio, valid_ratio, random_state, input_x_dim, input_y_dim, data_type)

    # Load Best Models
    model.load_state_dict(bestParams_picnn)
    model = model.to(device)
    # Obtain test metrics numbers
    x_test = testData[:, input_y_dim:].requires_grad_(True).to(device)
    y_test = testData[:, :input_y_dim].requires_grad_(True).to(device)
    log_prob_picnn = model.loglik_picnn(x_test, y_test)
    pb_mean_NLL = -log_prob_picnn.mean()
    # Calculate MMD
    zx = torch.randn(testData.shape[0], input_x_dim).to(device)
    x_generated, _ = model.gx(zx, testData[:, :input_y_dim].to(device), tol=tol)
    x_generated = x_generated.detach().to(device)
    mean_max_dis = mmd(x_generated, testData[:, input_y_dim:])

    return pb_mean_NLL.item(), mean_max_dis


"""
Training Process
"""

if __name__ == '__main__':

    """Load Data"""

    if args.data_type == 'real':
        data_path = f'ens/{args.data}.p'
    else:
        data_path = f'ensembles_a=[0.2,1.5]/ens_{args.data}.npy'

    data = np.load(data_path, allow_pickle=True)

    train_loader, valid_loader, n_train = load_data(data, args.test_ratio, args.valid_ratio,
                                                    args.batch_size, args.random_state, args.pca_components_s, args.pca_components_y, args.data_type) 

    """Construct Model"""
    reparam = not args.clip

    # Multivariate Gaussian as Reference
    input_y_dim = args.pca_components_y
    if args.data_type == 'real':
        input_x_dim = args.pca_components_s + 2
    else:
        input_x_dim = args.pca_components_s
    prior_picnn = distributions.MultivariateNormal(torch.zeros(input_x_dim).to(device), torch.eye(input_x_dim).to(device))

    # build PCP-Map
    picnn = PICNN(input_x_dim, input_y_dim, args.feature_dim, args.feature_y_dim,
                  1, args.num_layers_pi, reparam=reparam)
    pcpmap = PCPMap(prior_picnn, picnn).to(device)

    optimizer = torch.optim.Adam(pcpmap.parameters(), lr=args.lr)

    """Initial Logs"""

    data_filename = os.path.basename(data_path)
    data_filename = os.path.splitext(data_filename)[0]

    strTitle = data_filename + '_' + sStartTime + '_' + str(args.batch_size) + '_' + str(args.lr) + \
            '_' + str(args.num_layers_pi) + '_' + str(args.feature_dim)

    logger.info("--------------------------------------------------")
    logger.info("Number of trainable parameters: {}".format(count_parameters(picnn)))
    logger.info("--------------------------------------------------")
    logger.info(str(optimizer))  # optimizer info
    logger.info("--------------------------------------------------")
    logger.info("device={:}".format(device))
    logger.info("saveLocation = {:}".format(args.save))
    logger.info("--------------------------------------------------\n")

    columns_train = ["epoch", "step", "time/trnstep", "train_loss_p"]
    columns_valid = ["time/vldstep", "valid_loss_p"]
    train_hist = pd.DataFrame(columns=columns_train)
    valid_hist = pd.DataFrame(columns=columns_valid)

    logger.info(["iter"] + columns_train)

    """Training Starts"""

    # starts training
    itr = 1
    total_itr = (int(n_train / args.batch_size) + 1) * args.num_epochs
    best_loss_picnn = float('inf')
    bestParams_picnn = None

    makedirs(args.save)
    timeMeter = AverageMeter()
    vldTotTimeMeter = AverageMeter()

    for epoch in range(args.num_epochs):
        for i, sample in enumerate(train_loader):
            y = sample[:,:input_y_dim].requires_grad_(True).to(device)
            x = sample[:,input_y_dim:].requires_grad_(True).to(device)

            # start timer
            end = time.time()

            optimizer.zero_grad()
            loss = -pcpmap.loglik_picnn(x, y).mean()
            loss.backward()
            optimizer.step()

            # non-negative constraint
            if args.clip is True:
                for lw in pcpmap.picnn.Lw:
                    with torch.no_grad():
                        lw.weight.data = pcpmap.picnn.nonneg(lw.weight)

            # end timer
            step_time = time.time() - end
            timeMeter.update(step_time)
            train_hist.loc[len(train_hist.index)] = [epoch + 1, i + 1, step_time, loss.item()]

            # printing
            if itr % args.print_freq == 0:
                log_message = (
                    '{:05d}  {:7.1f}     {:04d}    {:9.3e}      {:9.3e} '.format(
                        itr, epoch + 1, i + 1, step_time, loss.item()
                    )
                )
                logger.info(log_message)

            if itr % args.valid_freq == 0 or itr == total_itr:
                vldtimeMeter = AverageMeter()
                valLossMeterPICNN = AverageMeter()
                for valid_sample in valid_loader:
                    y_valid = valid_sample[:,:input_y_dim].requires_grad_(True).to(device)
                    x_valid = valid_sample[:,input_y_dim:].requires_grad_(True).to(device)

                    # start timer
                    end_vld = time.time()
                    mean_valid_loss_picnn = -pcpmap.loglik_picnn(x_valid, y_valid).mean()
                    # end timer
                    batch_step_time = time.time() - end_vld
                    vldtimeMeter.update(batch_step_time)
                    valLossMeterPICNN.update(mean_valid_loss_picnn.item(), valid_sample.shape[0])

                val_loss_picnn = valLossMeterPICNN.avg
                vldstep_time = vldtimeMeter.sum
                vldTotTimeMeter.update(vldstep_time)

                valid_hist.loc[len(valid_hist.index)] = [vldstep_time, val_loss_picnn]
                log_message_valid = '   {:9.3e}      {:9.3e} '.format(vldstep_time, val_loss_picnn)

                if val_loss_picnn < best_loss_picnn:
                    n_vals_wo_improve_picnn = 0
                    best_loss_picnn = val_loss_picnn
                    makedirs(args.save)
                    bestParams_picnn = pcpmap.state_dict()
                    torch.save({
                        'args': args,
                        'state_dict_picnn': bestParams_picnn,
                    }, os.path.join(args.save, strTitle + '_checkpt.pth'))
                else:
                    n_vals_wo_improve_picnn += 1
                log_message_valid += '    picnn no improve: {:d}/{:d}'.format(n_vals_wo_improve_picnn,
                                                                              args.early_stopping)

                logger.info(columns_valid)
                logger.info(log_message_valid)
                logger.info(["iter"] + columns_train)

            # update learning rate
            if n_vals_wo_improve_picnn > args.early_stopping:
                if ndecs_picnn > 2:
                    logger.info("early stopping engaged")
                    logger.info("Training Time: {:} seconds".format(timeMeter.sum))
                    logger.info("Validation Time: {:} seconds".format(vldTotTimeMeter.sum))
                    train_hist.to_csv(os.path.join(args.save, '%s_train_hist.csv' % strTitle))
                    valid_hist.to_csv(os.path.join(args.save, '%s_valid_hist.csv' % strTitle))
                    if bool(args.save_test) is False:
                        exit(0)
                    else:
                        data = np.load(data_path, allow_pickle=True)
                        if args.data_type == 'real':
                            input_x_dim = args.pca_components_s + 2
                        else:
                            input_x_dim = args.pca_components_s
                        NLL, MMD = evaluate_model(pcpmap, data, args.batch_size, args.test_ratio, args.valid_ratio,
                                                  args.random_state, args.pca_components_y, input_x_dim, args.tol,
                                                  bestParams_picnn)

                        columns_test = ["batch_size", "lr", "width", "width_y", "depth", "NLL", "MMD", "time", "iter"]
                        test_hist = pd.DataFrame(columns=columns_test)
                        test_hist.loc[len(test_hist.index)] = [args.batch_size, args.lr, args.feature_dim,
                                                               args.feature_y_dim,
                                                               args.num_layers_pi, NLL, MMD, timeMeter.sum, itr]
                        data_filename = os.path.basename(data_path)
                        data_filename = os.path.splitext(data_filename)[0]
                        testfile_name = os.path.join(args.save, f'{data_filename}_test_hist.csv')

                        if os.path.isfile(testfile_name):
                            test_hist.to_csv(testfile_name, mode='a', index=False, header=False)
                        else:
                            test_hist.to_csv(testfile_name, index=False)
                        exit(0)
                else:
                    update_lr_picnn(optimizer, n_vals_wo_improve_picnn)
                    n_vals_wo_improve_picnn = 0

            itr += 1

    print('Training time: %.2f secs' % timeMeter.sum)
    print('Validation time: %.2f secs' % vldTotTimeMeter.sum)
    train_hist.to_csv(os.path.join(args.save, '%s_train_hist.csv' % strTitle))
    valid_hist.to_csv(os.path.join(args.save, '%s_valid_hist.csv' % strTitle))
    if bool(args.save_test) is False:
        exit(0)
    else:
        data = np.load(data_path, allow_pickle=True)
        if args.data_type == 'real':
            input_x_dim = args.pca_components_s + 2
        else:
            input_x_dim = args.pca_components_s
        NLL, MMD = evaluate_model(pcpmap, data, args.batch_size, args.test_ratio, args.valid_ratio,
                                  args.random_state, args.pca_components_y, input_x_dim, args.tol, bestParams_picnn)

        columns_test = ["batch_size", "lr", "width", "width_y", "depth", "NLL", "MMD", "time", "iter"]
        test_hist = pd.DataFrame(columns=columns_test)
        test_hist.loc[len(test_hist.index)] = [args.batch_size, args.lr, args.feature_dim, args.feature_y_dim,
                                               args.num_layers_pi, NLL, MMD,
                                               timeMeter.sum, itr]
        
        data_filename = os.path.basename(data_path)
        data_filename = os.path.splitext(data_filename)[0]
        testfile_name = os.path.join(args.save, f'{data_filename}_test_hist.csv')

        if os.path.isfile(testfile_name):
            test_hist.to_csv(testfile_name, mode='a', index=False, header=False)
        else:
            test_hist.to_csv(testfile_name, index=False)
