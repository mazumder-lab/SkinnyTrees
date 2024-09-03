#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4-00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=shibal@mit.edu
#SBATCH --output=sparse_average_%j_%a.out
#SBATCH --error=sparse_average_%j_%a.err
#SBATCH -a 0-9

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

# # churn
# /home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'churn' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --batch_size_scaler 1 --constant_learning_rate 0.0890487196819779 --num_trees 45 --depth 3 --epochs 180 --kernel_l2 0.5557920251351604 --kernel_constraint 100 --anneal --temperature 0.0156759025968044 --n_trials 10 --use_passed_hyperparameters --version 100 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_trees/skinny_trees/publicdata

# # satimage
# /home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'satimage' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --anneal --max_trees 100 --max_depth 6 --max_epochs 500 --n_trials 10 --use_passed_hyperparameters --version 100 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_trees/skinny_trees/publicdata

# # texture
# /home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'texture' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --anneal --max_trees 100 --max_depth 6 --max_epochs 500 --n_trials 10 --use_passed_hyperparameters --version 100 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_trees/skinny_trees/publicdata

# # mice-protein
# /home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'mice-protein' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --anneal --max_trees 100 --max_depth 6 --max_epochs 500 --n_trials 10 --use_passed_hyperparameters --version 100 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_trees/skinny_trees/publicdata


# # isolet
# /home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'isolet' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --anneal --max_trees 100 --max_depth 6 --max_epochs 500 --n_trials 10 --use_passed_hyperparameters --version 100 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_trees/skinny_trees/publicdata

# # lung
# /home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'lung' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --anneal --max_trees 100 --max_depth 6 --max_epochs 1000 --n_trials 10 --use_passed_hyperparameters --version 100 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_trees/skinny_trees/publicdata

# # tox
# /home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'tox' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --anneal --max_trees 100 --max_depth 6 --max_epochs 1000 --n_trials 10 --use_passed_hyperparameters --version 100 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_trees/skinny_trees/publicdata

# # cll
# /home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'cll' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --anneal --max_trees 100 --max_depth 6 --max_epochs 1000 --n_trials 10 --use_passed_hyperparameters --version 100 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_trees/skinny_trees/publicdata

# # smk
# /home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'smk' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --batch_size_scaler 1 --constant_learning_rate 0.3762418616036114 --num_trees 12 --depth 2 --epochs 270 --kernel_l2 0.9814409479011488 --kernel_constraint 100 --anneal --temperature 0.003272923926248 --n_trials 10 --use_passed_hyperparameters --version 100 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_trees/skinny_trees/publicdata

# # gli
# /home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'gli' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --batch_size_scaler 1 --constant_learning_rate 0.3811918761314004 --num_trees 1 --depth 1 --epochs 790 --kernel_l2 0.0345935808534042 --kernel_constraint 100 --anneal --temperature 0.0389778916028475 --n_trials 10 --use_passed_hyperparameters --version 100 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_trees/skinny_trees/publicdata

# # madelon
# /home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'madelon' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --batch_size_scaler 4 --constant_learning_rate 7.003012284492522 --num_trees 10 --depth 4 --epochs 120 --kernel_l2 0.0044092381218486 --kernel_constraint 100 --anneal --temperature 0.0194632884193298 --n_trials 10 --use_passed_hyperparameters --version 100 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_trees/skinny_trees/publicdata 

# # gisette
# /home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'gisette' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --batch_size_scaler 1 --constant_learning_rate 2.006151693256433 --num_trees 60 --depth 4 --epochs 955 --kernel_l2 0.0052262603894369 --kernel_constraint 100 --anneal --temperature 0.0007882874774981 --n_trials 10 --use_passed_hyperparameters --version 100 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_trees/skinny_trees/publicdata

# # dorothea
# /home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'dorothea' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --batch_size_scaler 4 --constant_learning_rate 8.589512894663638 --num_trees 54 --depth 4 --epochs 75 --kernel_l2 0.0063872944941386 --kernel_constraint 100 --anneal --temperature 0.0031597120213268 --n_trials 10 --use_passed_hyperparameters --version 100 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_trees/skinny_trees/publicdata

# # arcene
/home/gridsan/shibal/.conda/envs/MOETF29/bin/python /home/gridsan/shibal/SkinnyTrees/scripts/main_classification_public_data.py --data 'arcene' --data_type 'classification' --load_directory /home/gridsan/shibal/public-datasets --seed 8 --batch_size_scaler 4 --constant_learning_rate 1.187201190659109 --num_trees 11 --depth 2 --epochs 980 --kernel_l2 0.082868279738126 --kernel_constraint 100 --anneal --temperature 0.0036478197843367 --n_trials 10 --use_passed_hyperparameters --version 100 --tuning_seed $SLURM_ARRAY_TASK_ID --loss 'cross-entropy' --save_directory ./logs_trees/skinny_trees/publicdata
