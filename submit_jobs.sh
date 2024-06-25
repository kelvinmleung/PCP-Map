#!/bin/bash

# Source necessary profiles and load modules
source /etc/profile
module load anaconda/2023a

# Parse arguments
data=$1
data_type=$2

# Define the path for your pretrain script
pretrain_script='/home/gridsan/xzheng/code/PCP-Map/pretrain_cond.py'
experiment_script='/home/gridsan/xzheng/code/PCP-Map/experiment.py'

# Check if pretrain script exists
if [ ! -f "$pretrain_script" ]; then
    echo "Error: $pretrain_script not found."
    exit 1
fi

# Pretrain step
python "$pretrain_script" --data "$data" --data_type "$data_type"

# Check if experiment script exists
if [ ! -f "$experiment_script" ]; then
    echo "Error: $experiment_script not found."
    exit 1
fi

# Run the experiment script
python "$experiment_script" "$data" "$data_type"
