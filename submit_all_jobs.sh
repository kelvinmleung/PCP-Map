#!/bin/bash

# Usage: ./submit_all_jobs.sh

# Loading the required module
source /etc/profile
module load anaconda/2023a

# Define arrays for data and data_type
data_list=('177' '306' 'dark' 'mars')
data_type_list=('real' 'synthetic')

# Create logs directory if it doesn't exist
mkdir -p /home/gridsan/xzheng/code/PCP-Map/logs

# Loop through each combination of data and data_type
for data in "${data_list[@]}"; do
    for data_type in "${data_type_list[@]}"; do
        # Create a unique job script for each combination
        job_script=$(mktemp /tmp/job_${data}_${data_type}.XXXXXX.sh)
        
        # Write the job script
        cat << EOF > $job_script
#!/bin/bash
#LLsub -n 1
#LLsub -o /home/gridsan/xzheng/code/PCP-Map/logs/${data}_${data_type}_%J.out
#LLsub -e /home/gridsan/xzheng/code/PCP-Map/logs/${data}_${data_type}_%J.err

# Source necessary profiles and load modules
source /etc/profile
module load anaconda/2023a

# Define the paths for your Python scripts
pretrain_script='/home/gridsan/xzheng/code/PCP-Map/pretrain_cond.py'
experiment_script='/home/gridsan/xzheng/code/PCP-Map/experiment.py'

# Check if pretrain script exists
if [ ! -f "\$pretrain_script" ]; then
    echo "Error: \$pretrain_script not found."
    exit 1
fi

# Pretrain step
python "\$pretrain_script" --data "$data" --data_type "$data_type"

# Check if experiment script exists
if [ ! -f "\$experiment_script" ]; then
    echo "Error: \$experiment_script not found."
    exit 1
fi

# Run the experiment script
python "\$experiment_script" "$data" "$data_type"
EOF

        # Make the job script executable
        chmod +x $job_script

        # Submit the job script using LLsub
        LLsub $job_script

        # Optional: Clean up temporary job script
        # rm $job_script
    done
done
