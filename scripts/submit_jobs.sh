#!/bin/bash
# SBATCH --cpus-per-task=4
# SBATCH --mem=16G
# SBATCH --time=4-00:00
# SBATCH --mail-type=FAIL
# SBATCH --mail-user=shibal@mit.edu
# SBATCH --output=sparse_%j_%a.out
# SBATCH --error=sparse_%j_%a.err
# SBATCH -a 0-7

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

# churn
/home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'churn' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --anneal --max_trees 100 --max_depth 6 --max_epochs 500 --n_trials 250 --version 1 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_skinny_trees/soft_trees/publicdata

# satimage
/home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'satimage' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --anneal --max_trees 100 --max_depth 6 --max_epochs 500 --n_trials 250 --version 1 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_skinny_trees/soft_trees/publicdata

# texture
/home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'texture' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --anneal --max_trees 100 --max_depth 6 --max_epochs 500 --n_trials 250 --version 1 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_skinny_trees/soft_trees/publicdata

# mice-protein
/home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'mice-protein' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --anneal --max_trees 100 --max_depth 6 --max_epochs 500 --n_trials 250 --version 1 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_skinny_trees/soft_trees/publicdata


# isolet
/home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'isolet' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --anneal --max_trees 100 --max_depth 6 --max_epochs 500 --n_trials 250 --version 1 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_skinny_trees/soft_trees/publicdata

# lung
/home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'lung' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --anneal --max_trees 100 --max_depth 6 --max_epochs 1000 --n_trials 500 --version 1 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_skinny_trees/soft_trees/publicdata

# tox
/home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'tox' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --anneal --max_trees 100 --max_depth 6 --max_epochs 1000 --n_trials 500 --version 1 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_skinny_trees/soft_trees/publicdata

# smk
/home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'smk' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --anneal --max_trees 100 --max_depth 6 --max_epochs 1000 --n_trials 250 --version 1 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_skinny_trees/soft_trees/publicdata

# gli
/home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'gli' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --anneal --max_trees 100 --max_depth 6 --max_epochs 1000 --n_trials 250 --version 1 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_skinny_trees/soft_trees/publicdata

# cll
/home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'cll' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --anneal --max_trees 100 --max_depth 6 --max_epochs 1000 --n_trials 250 --version 1 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_skinny_trees/soft_trees/publicdata

# madelon
/home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'madelon' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --anneal --max_trees 100 --max_depth 6 --max_epochs 1000 --n_trials 250 --version 1 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_skinny_trees/soft_trees/publicdata 

# gisette
/home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'gisette' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --anneal --max_trees 100 --max_depth 6 --max_epochs 1000 --n_trials 125 --version 1 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_skinny_trees/soft_trees/publicdata

# dorothea
/home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'dorothea' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --anneal --max_trees 100 --max_depth 6 --max_epochs 1000 --n_trials 50 --version 1 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_skinny_trees/soft_trees/publicdata

# arcene
/home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'arcene' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --anneal --max_trees 100 --max_depth 6 --max_epochs 1000 --n_trials 200 --version 1 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_skinny_trees/soft_trees/publicdata
