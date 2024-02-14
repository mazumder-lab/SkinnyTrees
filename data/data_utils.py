"""Data Processing Utilities
"""
import collections
import copy
import numpy as np
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import yaml
import scipy.io

def load_processed_classification_public_data(
    name="human-activity-recognition",
    train_size=None,
    val_size=0.2,
    test_size=0.2,
    seed=8,
    path="s3://cortex-mit1003-lmdl-workbucket/public-datasets-processed",
    ):
    
    if name in ["mice-protein", "isolet"]:
        path = os.path.join(path, "singletask-datasets/fetch-openml-datasets/classification")
        df_X = pd.read_csv(os.path.join(path, "{}/features.csv".format(name)))
        df_y = pd.read_csv(os.path.join(path, "{}/target.csv".format(name)))
        df_y = df_y['target']
    elif name in ['churn', 'satimage','texture']:
        path = os.path.join(path, "singletask-datasets/pmlb-datasets/classification")
        df = pd.read_csv(path+"/{}/{}.tsv".format(name.replace("-", "_"), name.replace("-", "_")), sep="\t")
        df_y = df['target']
        df_X = df.drop(columns='target')
    elif name in ['arcene', 'gisette', 'madelon', 'dorothea']:
        path = os.path.join(path, "feature-selection-datasets/nips2003")
        data =  scipy.io.loadmat(path+'/{}/{}/{}.mat'.format(name.upper(), name.upper(), name))
        x_train_valid = pd.DataFrame(data['Xtrain'])
        y_train_valid = pd.DataFrame(data['ytrain'], columns=['target'])
        y_train_valid = y_train_valid['target']
        x_test = pd.DataFrame(data['Xvalid'])
        y_test = pd.DataFrame(data['yvalid'], columns=['target'])
        y_test = y_test['target']
    elif name in ['smk', 'gli', 'cll', 'lung', 'tox']:
        if name in ['basehock', 'pcmac', 'relathe']:
            path = os.path.join(path, "feature-selection-datasets/text")
        elif name in ['smk', 'gli', 'cll', 'lung', 'tox']:
            path = os.path.join(path, "feature-selection-datasets/bio")
        data =  scipy.io.loadmat(path+'/{}.mat'.format(name.upper()))
        df_X = pd.DataFrame(data['X'])
        df_y = pd.DataFrame(data['Y'].reshape(-1), columns=['target'])
        df_y = df_y['target']
        
    else:
        raise ValueError("Data: '{}' is not supported".format(name))
    
    if name not in ['arcene', 'gisette', 'madelon', 'dorothea']:
        classes = list(set(df_y.values))
        print("classes:", classes)

        print("X.min:", df_X.min().sort_values(), "X.max:", df_X.max().sort_values())
        print("name:", name, "X.shape:", df_X.shape, "y.shape:", df_y.shape)

        np.random.seed(seed)
        x_train_valid, x_test, y_train_valid, y_test = train_test_split(df_X, df_y, test_size=test_size, stratify=df_y, random_state=seed)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid, y_train_valid, test_size=val_size, stratify=y_train_valid, random_state=seed)
        print(x_train.nunique())
    else:
        classes = list(set(y_train_valid.values))
        print("classes:", classes)

        print("X.min:", x_train_valid.min().sort_values(), "X.max:", x_train_valid.max().sort_values())
        print("name:", name, "X.shape:", x_train_valid.shape, "y.shape:", y_train_valid.shape)

        np.random.seed(seed)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid, y_train_valid, test_size=val_size, stratify=y_train_valid, random_state=seed)
        print(x_train.nunique())
        
        
    print("Number of samples in training set: ", x_train.shape[0], y_train.shape[0])
    print("Number of samples in validation set: ", x_valid.shape[0], y_valid.shape[0])
    print("Number of samples in train+validation set: ", x_train_valid.shape[0], y_train_valid.shape[0])
    print("Number of samples in testing set: ", x_test.shape[0], y_test.shape[0])
    print("Percentage of missing vals in training covariates: ", 100*np.count_nonzero(x_train.isna().values)/(x_train.values.size))
    print("Percentage of missing vals in validation covariates: ", 100*np.count_nonzero(x_valid.isna().values)/(x_valid.values.size))
    print("Percentage of missing vals in train+validation covariates: ", 100*np.count_nonzero(x_train_valid.isna().values)/(x_train_valid.values.size))
    print("Percentage of missing vals in testing covariates: ", 100*np.count_nonzero(x_test.isna().values)/(x_test.values.size))
    print("Number of NaNs in tasks responses in training set: ", y_train.isna().values.sum(axis=0))
    print("Number of NaNs in tasks responses in validation set: ", y_valid.isna().values.sum(axis=0))
    print("Number of NaNs in tasks responses in train+validation set: ", y_train_valid.isna().values.sum(axis=0))
    print("Number of NaNs in tasks responses in train+validation set: ", y_test.isna().values.sum(axis=0))
    
    print(x_train.shape, x_valid.shape, x_train_valid.shape, x_test.shape)
    print(y_train.shape, y_valid.shape, y_train_valid.shape, y_test.shape)
    
    if name in ['mice-protein', 'isolet']:
        metadata = {
            'continuous_features': df_X.columns,
            'categorical_features': [],
            'binary_features': [],
            'ordinal_features': [],
            'nominal_features': [],
        }
    elif name in ['churn', 'satimage', 'texture']:
        with open(os.path.join(path, "{}/metadata.yaml".format(name.replace("-", "_"))), "r") as stream:
            try:
                metadata = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        df_metadata = pd.DataFrame(metadata['features'])
        from IPython.display import display
        display(df_metadata)
        metadata = {
            'continuous_features': df_metadata[df_metadata['type']=='continuous'].name.astype(str).values,
            'categorical_features': df_metadata[df_metadata['type']=='categorical'].name.astype(str).values,
            'binary_features': df_metadata[df_metadata['type']=='binary'].name.astype(str).values,
            'nominal_features': df_metadata[df_metadata['type']=='nominal'].name.astype(str).values,
            'ordinal_features': df_metadata[df_metadata['type']=='ordinal'].name.astype(str).values,
        }
        print(metadata['ordinal_features'])
    elif name in ['arcene', 'gisette', 'madelon', 'dorothea', 'smk', 'gli', 'cll', 'lung', 'tox']:
        if name=='gisette':
            metadata = {
                'continuous_features': x_train_valid.columns,
                'categorical_features': [],
                'binary_features': [],
                'nominal_features': [],
                'ordinal_features': [],
            }
        elif name=='madelon':
            metadata = {
                'continuous_features': x_train_valid.columns,
                'categorical_features': [],
                'binary_features': [],
                'nominal_features': [],
                'ordinal_features': [],
            }
        elif name=='arcene':
            metadata = {
                'continuous_features': x_train_valid.columns,
                'categorical_features': [],
                'binary_features': [],
                'nominal_features': [],
                'ordinal_features': [],
            }
        elif name=='dorothea':
            metadata = {
                'continuous_features': [],
                'categorical_features': [],
                'binary_features': x_train_valid.columns,
                'nominal_features': [],
                'ordinal_features': [],
            }
        elif name in ['smk', 'gli', 'cll', 'lung', 'tox']:
            metadata = {
                'continuous_features': x_train_valid.columns,
                'categorical_features': [],
                'binary_features': [],
                'nominal_features': [],
                'ordinal_features': [],
            }
            
    if name not in ['arcene', 'gisette', 'madelon', 'dorothea', 'smk', 'gli', 'cll', 'lung', 'tox']:
        df_X[metadata['ordinal_features']] = df_X[metadata['ordinal_features']].apply(pd.to_numeric)
        df_X[metadata['continuous_features']] = df_X[metadata['continuous_features']].apply(pd.to_numeric)
        print(df_X.shape)
        
    
    if name in ['mice-protein', 'isolet', 'churn', 'satimage','texture',
                'arcene','gisette', 'madelon', 'smk', 'gli', 'cll', 'lung', 'tox']:
        continuous_features = metadata['continuous_features']
        continuous_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_features =  metadata['categorical_features']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        binary_features =  metadata['binary_features']
        binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        nominal_features =  metadata['nominal_features']
        nominal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        ordinal_features = metadata['ordinal_features']
        ordinal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
        
        x_preprocessor = ColumnTransformer(
            transformers=[
                ('continuous', continuous_transformer, continuous_features),
                ('categorical', categorical_transformer, categorical_features),
                ('binary', binary_transformer, binary_features),
                ('nominal', nominal_transformer, nominal_features),
                ('ordinal', ordinal_transformer, ordinal_features),                
            ])

        print(x_train.nunique().sort_values())
        x_train_processed = x_preprocessor.fit_transform(x_train)
        x_valid_processed = x_preprocessor.transform(x_valid)
        x_train_valid_processed = x_preprocessor.transform(x_train_valid)    
        x_test_processed = x_preprocessor.transform(x_test)
    elif name in ['dorothea']:
        x_train_processed = x_train.values
        x_valid_processed = x_valid.values
        x_train_valid_processed = x_train_valid.values    
        x_test_processed = x_test.values
        

    y_preprocessor = LabelEncoder()
    y_train_processed = y_preprocessor.fit_transform(y_train)
    y_valid_processed = y_preprocessor.transform(y_valid)
    y_train_valid_processed = y_preprocessor.transform(y_train_valid)
    y_test_processed = y_preprocessor.transform(y_test)
    
    print(x_train_processed.shape, x_valid_processed.shape, x_train_valid_processed.shape, x_test_processed.shape)
    print(y_train_processed.shape, y_valid_processed.shape, y_train_valid_processed.shape, y_test_processed.shape)
    data_coll = collections.namedtuple('data', ['x_train', 'x_valid', 'x_train_valid', 'x_test',
                                                'y_train', 'y_valid', 'y_train_valid', 'y_test',
                                                'x_train_processed', 'x_valid_processed',
                                                'x_train_valid_processed', 'x_test_processed',
                                                'y_train_processed', 'y_valid_processed',
                                                'y_train_valid_processed', 'y_test_processed'])
    data_processed = data_coll(x_train, x_valid, x_train_valid, x_test,
                               y_train, y_valid, y_train_valid, y_test,
                               x_train_processed, x_valid_processed,
                               x_train_valid_processed, x_test_processed,
                               y_train_processed, y_valid_processed, y_train_valid_processed, y_test_processed)
    return data_processed

