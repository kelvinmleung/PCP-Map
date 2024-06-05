import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def dataloader(data, batch_size, test_ratio, valid_ratio, random_state, pca_components_x, pca_components_y):
    """
    :param dataset: dataset to load
    :param batch_size: batch size for Dataloader
    :param test_ratio: ratio for test set
    :param valid_ratio: ratio for validation set
    :param random_state: random seed for shuffling
    :return: Dataloaders for train, test, validation set
    """
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
    train_valid, test = train_test_split(
            data, test_size=test_ratio,
            random_state=random_state
        )
    nTot = data.shape[0]
    train, valid = train_test_split(
        train_valid, test_size=(nTot * valid_ratio) / train_valid.shape[0],
        random_state=random_state
    )
    train_mean = np.mean(train, axis=0, keepdims=True)
    train_std = np.std(train, axis=0, keepdims=True)
    train = (train - train_mean) / train_std
    valid = (valid - train_mean) / train_std
    test = (test - train_mean) / train_std

    # convert to tensor
    train_data = torch.tensor(train, dtype=torch.float32)
    valid_data = torch.tensor(valid, dtype=torch.float32)
    test_data = torch.tensor(test, dtype=torch.float32)

    train_sz = train.shape[0]

    # load train data
    trn_loader = DataLoader(
        train_data,
        batch_size=batch_size, shuffle=True
    )
    vld_loader = DataLoader(
        valid_data,
        batch_size=batch_size, shuffle=True
    )

    # load train data
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size, shuffle=True
    )


    return train_loader, valid_loader, test_data, train_sz