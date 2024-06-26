#!/bin/bash
# SBATCH --job-name=test_pcp_map       # Job name
# SBATCH --output=run_trial_output.txt # Standard output file
# SBATCH --error=run_trial_error.txt   # Standard error file

source /etc/profile
module load anaconda/2023a

#Your job commands go here
python pretrain_cond.py --data '177' --data_type 'real'