#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4-00:00
#SBATCH --output=sparse_synthetic_%j_%a.out
#SBATCH --error=sparse_synthetic_%j_%a.err
#SBATCH -a 1-25

echo 'My SLURM_ARRAY_TASK_ID: ' $SLURM_ARRAY_TASK_ID
echo 'Number of Tasks: ' $SLURM_ARRAY_TASK_COUNT

# Initialize the module command first
source /etc/profile

# Load modules
module load anaconda/2022b
export LD_LIBRARY_PATH=/state/partition1/llgrid/pkg/anaconda/2022b/lib:$LD_LIBRARY_PATH
# module load cuda/11.5
# module load anaconda/2023a-tensorflow
# export LD_LIBRARY_PATH=/state/partition1/llgrid/pkg/anaconda/2023a-tensorflow/lib:$LD_LIBRARY_PATH

# Call your script as you would from your command line
source activate MOETF29

export HDF5_USE_FILE_LOCKING=FALSE

cd /home/gridsan/shibal/SkinnyTrees/scripts
ulimit -u 32768

# # synthetic sigma=0.7 num_features=512 train_size=100
# /home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_regression_synthetic_data.py --data 'synthetic' --data_type 'regression' --seed $SLURM_ARRAY_TASK_ID --num_features 512 --sigma 0.7 --train_size 100 --test_size 10000 --anneal --max_trees 50 --max_depth 5 --max_epochs 500 --n_trials 500 --version 2 --tuning_seed 0 --loss 'mse' --save_directory ./logs_trees/skinny_trees/syntheticdata

# # synthetic sigma=0.7 num_features=512 train_size=200
# /home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_regression_synthetic_data.py --data 'synthetic' --data_type 'regression' --seed $SLURM_ARRAY_TASK_ID --num_features 512 --sigma 0.7 --train_size 200 --test_size 10000 --anneal --max_trees 50 --max_depth 5 --max_epochs 500 --n_trials 500 --version 2 --tuning_seed 0 --loss 'mse' --save_directory ./logs_trees/skinny_trees/syntheticdata

# # synthetic sigma=0.7 num_features=512 train_size=1000
# /home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_regression_synthetic_data.py --data 'synthetic' --data_type 'regression' --seed $SLURM_ARRAY_TASK_ID --num_features 512 --sigma 0.7 --train_size 1000 --test_size 10000 --anneal --max_trees 50 --max_depth 5 --max_epochs 500 --n_trials 500 --version 2 --tuning_seed 0 --loss 'mse' --save_directory ./logs_trees/skinny_trees/syntheticdata

# # synthetic sigma=0.5 num_features=256 train_size=100
# /home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_regression_synthetic_data.py --data 'synthetic' --data_type 'regression' --seed $SLURM_ARRAY_TASK_ID --num_features 256 --sigma 0.5 --train_size 100 --test_size 10000 --anneal --max_trees 50 --max_depth 5 --max_epochs 500 --n_trials 500 --version 2 --tuning_seed 0 --loss 'mse' --save_directory ./logs_trees/skinny_trees/syntheticdata

# # synthetic sigma=0.5 num_features=256 train_size=200
# /home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_regression_synthetic_data.py --data 'synthetic' --data_type 'regression' --seed $SLURM_ARRAY_TASK_ID --num_features 256 --sigma 0.5 --train_size 200 --test_size 10000 --anneal --max_trees 50 --max_depth 5 --max_epochs 500 --n_trials 500 --version 2 --tuning_seed 0 --loss 'mse' --save_directory ./logs_trees/skinny_trees/syntheticdata

# # synthetic sigma=0.5 num_features=256 train_size=1000
# /home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_regression_synthetic_data.py --data 'synthetic' --data_type 'regression' --seed $SLURM_ARRAY_TASK_ID --num_features 256 --sigma 0.5 --train_size 1000 --test_size 10000 --anneal --max_trees 50 --max_depth 5 --max_epochs 500 --n_trials 500 --version 2 --tuning_seed 0 --loss 'mse' --save_directory ./logs_trees/skinny_trees/syntheticdata


