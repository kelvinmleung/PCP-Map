import os
import sys
import pandas as pd
import numpy as np

def main(data, data_type):
    # Paths to the CSV files
    loss_file = f"{os.path.expanduser('~')}/code/PCP-Map/experiments/tabcond/ens_{data}_valid_hist.csv"
    param_file = f"{os.path.expanduser('~')}/code/PCP-Map/experiments/tabcond/ens_{data}_params_hist.csv"

    # Ensure loss_file and param_file exist
    if not os.path.exists(loss_file):
        print(f"Error: {loss_file} not found.")
        return 1

    if not os.path.exists(param_file):
        print(f"Error: {param_file} not found.")
        return 1

    # Read and process the CSV files to get the best hyperparameters
    loss = pd.read_csv(loss_file).to_numpy()
    param = pd.read_csv(param_file).to_numpy()
    loss_param = np.concatenate((param[:, 1:], loss[:, 1:]), axis=1)
    unique_param = loss_param[np.unique(loss_param[:, :-1], return_index=True, axis=0)[1]]
    unique_param = unique_param[unique_param[:, -1].argsort()]
    param_list = unique_param[0, :]

    batch_size = int(param_list[0])
    lr = param_list[1]
    width = int(param_list[2])
    width_y = int(param_list[3])
    num_layers = int(param_list[4])

    # Determine save directory based on data type
    if data_type == "real":
        save_dir = f"experiments/cond/{data}"
    else:
        save_dir = f"experiments/cond/ens_{data}"

    # Construct the command to run train_cond.py
    train_command = (
        f"python train_cond.py --data {data} --data_type {data_type} --valid_freq 50 --early_stopping 20 "
        f"--pca_components_s 40 --pca_components_y 40 --num_layers_pi {num_layers} --feature_dim {width} "
        f"--feature_y_dim {width_y} --batch_size {batch_size} --lr {lr} --save_test 1 --save {save_dir}"
    )

    # Execute the command
    os.system(train_command)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python experiment.py <data> <data_type>")
        sys.exit(1)
    data = sys.argv[1]
    data_type = sys.argv[2]
    sys.exit(main(data, data_type))
