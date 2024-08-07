import os
import numpy as np
import pandas as pd

# TODO change to correct paths
loss = pd.read_csv('~/code/PCP-Map/experiments/tabcond/ens_177_valid_hist.csv').to_numpy()
param = pd.read_csv('~/code/PCP-Map/experiments/tabcond/ens_177_params_hist.csv').to_numpy()
loss_param = np.concatenate((param[:, 1:], loss[:, 1:]), axis=1)
unique_param = loss_param[np.unique(loss_param[:, :-1], return_index=True, axis=0)[1]]
unique_param = unique_param[unique_param[:, -1].argsort()]
param_list = unique_param[0, :]

batch_size = int(param_list[0])
lr = param_list[1]
width = int(param_list[2])
width_y = int(param_list[3])
num_layers = int(param_list[4])

os.system(
    "python train_cond.py --data '177' --data_type 'synthetic' --valid_freq 50 --early_stopping 20 --pca_components_s 40 --pca_components_y 42"
    " --num_layers_pi " + str(num_layers) + " --feature_dim " + str(width) + " --feature_y_dim " + str(width_y) +
    " --batch_size " + str(batch_size) + " --lr " + str(lr) + " --save_test 1 --save 'experiments/cond/ens_177'"
)
