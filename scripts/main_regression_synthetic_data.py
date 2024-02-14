import os
import sys
import pandas as pd, numpy as np
import optuna
from optuna.samplers import RandomSampler
import timeit
import joblib
from copy import deepcopy
import pandas as pd
import time
import argparse
import collections

import pathlib
sys.path.insert(0, os.path.abspath(str(pathlib.Path(__file__).absolute()).split('src')[0]))

# from src import data_utils
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
# from sklearn.datasets import load_boston

from scripts import regression_tuning_synthetic_data


from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']

parser = argparse.ArgumentParser(description='Soft Tree Ensembles with feature selection.')

# Data Arguments
parser.add_argument('--load_directory', dest='load_directory',  type=str, default='s3://cortex-mit1003-lmdl-workbucket/public-datasets-processed')
parser.add_argument('--data', dest='data',  type=str, default='synthetic')
parser.add_argument('--data_type', dest='data_type',  type=str, default='regression')
parser.add_argument('--seed', dest='seed',  type=int, default=8)
parser.add_argument('--num_features', dest='num_features',  type=int, default=256)
parser.add_argument('--sigma', dest='sigma',  type=float, default=0.5)
parser.add_argument('--train_size', dest='train_size',  type=int, default=100)
parser.add_argument('--test_size', dest='test_size',  type=float, default=10000)

# Model Arguments
parser.add_argument('--anneal', action='store_true') # for dense-to-sparse

# Algorithm Arguments
parser.add_argument('--loss', dest='loss',  type=str, default='mse')
parser.add_argument('--max_trees', dest='max_trees',  type=int, default=100)
parser.add_argument('--max_depth', dest='max_depth',  type=int, default=4)
parser.add_argument('--max_epochs', dest='max_epochs',  type=int, default=200)
parser.add_argument('--n_trials', dest='n_trials',  type=int, default=2)
parser.add_argument('--patience', dest='patience',  type=int, default=25)

# Logging Arguments
parser.add_argument('--version', dest='version',  type=int, default=1)
parser.add_argument('--save_directory', dest='save_directory',  type=str, default='./logs/soft_trees/syntheticdata')

# Tuning Arguments
parser.add_argument('--tuning_criteria', dest='tuning_criteria',  type=str, default='mse')
parser.add_argument('--tuning_seed', dest='tuning_seed',  type=int, default=1)

args = parser.parse_args()

if args.data=='synthetic':
    p = args.num_features
    k = 8
    correlated = True
    if correlated:
        sigma = np.zeros((p,p))
        for i in range(p):
            for j in range(p):
                sigma[i,j] = args.sigma**(abs(i-j))
    else:
        sigma = np.eye(p)

    np.random.seed(args.seed)
    x_train_processed = np.random.multivariate_normal(np.zeros(p), sigma, (int)(0.8*args.train_size))
    x_valid_processed = np.random.multivariate_normal(np.zeros(p), sigma, (int)(0.2*args.train_size))
    x_test_processed = np.random.multivariate_normal(np.zeros(p), sigma, args.test_size)

    # scaler = StandardScaler()
    # x_train_processed = scaler.fit_transform(x_train)
    # x_valid_processed = scaler.transform(x_valid)
    # x_test_processed = scaler.transform(x_test)

    feature_support_truth = np.zeros(p)
    feature_support_truth[np.arange((int)(p/(2*k)),p,(int)(p/k))] = 1
    # feature_support_truth[np.random.choice(p,k,replace=False)] = 1

    ftrain = x_train_processed@feature_support_truth
    fval = x_valid_processed@feature_support_truth
    ftest = x_test_processed@feature_support_truth

    errortrain = np.random.normal(loc=0, scale=0.5, size=ftrain.shape)
    errorval = np.random.normal(loc=0, scale=0.5, size=fval.shape)
    errortest = np.random.normal(loc=0, scale=0.5, size=ftest.shape)

    y_train_processed = ftrain+errortrain
    y_valid_processed = fval+errorval
    y_test_processed = ftest+errortest

print("=============Dataset sizes===============")
print(x_train_processed.shape, x_valid_processed.shape, x_test_processed.shape)
print(y_train_processed.shape, y_valid_processed.shape, y_test_processed.shape)
print("min(x):", x_train_processed.min(), x_valid_processed.min(), x_test_processed.min())
print("max(x):", x_train_processed.max(), x_valid_processed.max(), x_test_processed.max())
print("min(y):", y_train_processed.min(), y_valid_processed.min(), y_test_processed.min())
print("max(y):", y_train_processed.max(), y_valid_processed.max(), y_test_processed.max())

data_coll = collections.namedtuple(
    'data', [
        'x_train_processed', 'x_valid_processed','x_test_processed',
        'y_train_processed', 'y_valid_processed','y_test_processed',
        'feature_support_truth'
    ]
)
data_processed = data_coll(
    x_train_processed, x_valid_processed, x_test_processed,
    y_train_processed, y_valid_processed, y_test_processed,
    feature_support_truth
)

path = os.path.join(
    args.save_directory,
    args.data,
    "{}".format(args.sigma),
    "{}".format(args.num_features),
    args.loss,
    "{}.{}".format(args.version, args.tuning_seed),
    "anneal{}".format(args.anneal),
    "train_size_{}".format(args.train_size),
    "results",
    "seed{}".format(args.seed)
)
os.makedirs(path, exist_ok=True)

config = dict({
    'early_stopping': False,
    'loss_criteria': args.loss
})
start = timeit.default_timer()

if args.loss in ['mse']:
    direction = 'minimize'
elif args.loss in ['cross-entropy']:
    direction = 'maximize'
study = optuna.create_study(sampler=RandomSampler(seed=args.tuning_seed), direction=direction)
objective_wrapper = lambda trial: regression_tuning_synthetic_data.objective(
    trial,
    data_processed,
    args.data_type,
    path,
    args,
    config,
)
study.optimize(objective_wrapper, n_trials=args.n_trials)

# best_params = study.best_params
# best_model=study.user_attrs["best_model"]

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
best_trial = study.best_trial

print("  Value: {}".format(best_trial.value))

print("  Params: ")
for key, value in best_trial.params.items():
    print("    {}: {}".format(key, value))

stop = timeit.default_timer()
hours, rem = divmod(stop-start, 3600)
minutes, seconds = divmod(rem, 60)
print("Training completed in {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds, args.n_trials)) 

df_study = study.trials_dataframe()
filename = "study"
# filename = filename + "-{}".format(args.max_trees)
# filename = filename + "-{}".format(args.max_depth)
if args.anneal:
    filename = filename + "-anneal"
# filename = filename + "_seed{}".format(args.seed)
filename = filename + ".csv"
df_study.to_csv(os.path.join(path, filename))

