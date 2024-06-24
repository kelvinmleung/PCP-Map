#!/bin/bash

# Source necessary profiles and load modules
source /etc/profile
module load anaconda/2023a

# Parse arguments
data=$1
data_type=$2

# Pretrain step
python pretrain.py --data "$data" --data_type "$data_type"

# Extract best hyperparameters and run training step
loss_file="~/code/PCP-Map/experiments/tabcond/ens_${data}_valid_hist.csv"
param_file="~/code/PCP-Map/experiments/tabcond/ens_${data}_params_hist.csv"
loss=$(python -c "import pandas as pd; print(pd.read_csv('${loss_file}').to_numpy())")
param=$(python -c "import pandas as pd; print(pd.read_csv('${param_file}').to_numpy())")
loss_param=$(python -c "import numpy as np; print(np.concatenate((${param}[:, 1:], ${loss}[:, 1:]), axis=1))")
unique_param=$(python -c "import numpy as np; up = ${loss_param}[np.unique(${loss_param}[:, :-1], return_index=True, axis=0)[1]]; print(up[up[:, -1].argsort()])")
param_list=$(python -c "print(${unique_param}[0, :])")

batch_size=$(python -c "print(int(${param_list}[0]))")
lr=$(python -c "print(${param_list}[1])")
width=$(python -c "print(int(${param_list}[2]))")
width_y=$(python -c "print(int(${param_list}[3]))")
num_layers=$(python -c "print(int(${param_list}[4]))")

# Determine save directory based on data type
if [ "$data_type" == "real" ]; then
    save_dir="experiments/cond/${data}"
else
    save_dir="experiments/cond/ens_${data}"
fi

# Run training step
python train.py --data "$data" --data_type "$data_type" --valid_freq 50 --early_stopping 20 --pca_components_s 40 --pca_components_y 40 \
    --num_layers_pi "$num_layers" --feature_dim "$width" --feature_y_dim "$width_y" \
    --batch_size "$batch_size" --lr "$lr" --save_test 1 --save "$save_dir"
