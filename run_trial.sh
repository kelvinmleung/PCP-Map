#!/bin/bash
# SBATCH --job-name=test_pcp_map       # Job name
# SBATCH --output=run_trial_output.txt # Standard output file
# SBATCH --error=run_trial_error.txt   # Standard error file

#Load necessary modules (if needed)
module load anaconda/2023a

#Your job commands go here
python pretrain_cond.py --data 'retrieval' --input_x_dim 40 --input_y_dim 40 --out_dim 1