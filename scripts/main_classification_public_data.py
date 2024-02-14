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

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib
from sklearn.metrics import accuracy_score, roc_auc_score

import pathlib

import pathlib
sys.path.insert(0, os.path.abspath(str(pathlib.Path(__file__).absolute()).split('src')[0]))

from data import data_utils
from src import models
from src import sparse_soft_trees


from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


def objective(
    trial,
    data_processed,
    data_type,
    path,
    args,
    config,
    use_passed_hyperparameters,
    ):
    
    """ Clear the backend (TensorFlow). See:
    https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session
    """
    K.clear_session() 
    
    global trial_nb
    trial_nb = trial_nb + 1


    assert len(data_processed.y_train_processed.shape)==1
    # print("Class-set (Train)", np.unique(data_processed.y_train_processed))
    # print("Class-set (Valid)", np.unique(data_processed.y_valid_processed))
    # print("Class-set (Train-Valid)", np.unique(data_processed.y_train_valid_processed))
    # print("Class-set (Test)", np.unique(data_processed.y_test_processed))
    if data_type=='classification':
        num_classes = data_processed.y_train_valid_processed.max(axis=0)+1
        print("====================num_classes", num_classes)
    print("====================path", path)
    num_features = data_processed.x_train_processed.shape[1]
    
    if not use_passed_hyperparameters:
        config.update({
            'constant_batch_size': 16, # trial.suggest_categorical('constant_batch_size', [16])
            'batch_size_scaler': trial.suggest_categorical('batch_size_scaler', [1,4,16,64]),
            'constant_learning_rate': trial.suggest_loguniform('constant_learning_rate', 1e-2, 1e1),
            'num_trees': trial.suggest_int('num_trees', 1, args.max_trees),
            'depth': trial.suggest_int('depth', 1, args.max_depth),
            'use_annealing': args.anneal, # trial.suggest_categorical('use_annealing', [args.anneal])
        })

        if config['early_stopping']:
            config['epochs'] = args.max_epochs
        else:
            config['epochs'] = trial.suggest_int('epochs', 5, args.max_epochs, 5) # [5,500] originally

        config['kernel_l2'] = trial.suggest_loguniform('kernel_l2', 1e-3, 1e0)
        if config['use_annealing']:
            config['kernel_constraint'] = 100.0 # trial.suggest_categorical('kernel_constraint', [100.0])
            config['temperature'] = trial.suggest_loguniform('temperature', 1e-4, 1e-1)
        else:
            config['kernel_constraint'] = trial.suggest_loguniform('kernel_constraint', 1e+0, 1e+4)
    else:
        config['constant_batch_size'] =  16 # trial.suggest_categorical('constant_batch_size', [16])
        config['batch_size_scaler'] = trial.suggest_categorical('constant_batch_size', [args.batch_size_scaler])
        config['constant_learning_rate'] = trial.suggest_categorical('constant_learning_rate', [args.constant_learning_rate])
        config['num_trees'] = trial.suggest_categorical('num_trees', [args.num_trees])
        config['depth'] = trial.suggest_categorical('depth', [args.depth])
        config['use_annealing'] = trial.suggest_categorical('use_annealing', [args.anneal]) # trial.suggest_categorical('use_annealing', [args.anneal])
        config['epochs'] = trial.suggest_categorical('epochs', [args.epochs])
        config['kernel_constraint'] = trial.suggest_categorical('kernel_constraint', [args.kernel_constraint])
        config['kernel_l2'] = trial.suggest_categorical('kernel_l2', [args.kernel_l2])
        config['kernel_regularizer'] = trial.suggest_categorical('kernel_regularizer', [args.kernel_regularizer])
        config['temperature'] = trial.suggest_categorical('temperature', [args.temperature])
#         config['cardinality'] = trial.suggest_categorical('cardinality', [(int)(np.ceil(0.25*num_features))])
        
            
    constant_batch_size = config['constant_batch_size']
    batch_size_scaler = config['batch_size_scaler']
    batch_size = constant_batch_size*batch_size_scaler
    constant_learning_rate = config['constant_learning_rate']
    early_stopping = config['early_stopping']

    
    num_train_samples = data_processed.x_train_processed.shape[0]
    epochs = config['epochs']
    
    if num_train_samples % batch_size == 0:
        epoch_step = num_train_samples / batch_size
    else:
        epoch_step = int(num_train_samples / batch_size) + 1
        
#         epochs = trial.suggest_int('epochs_new', 5, epochs)
    print("==============No LR scheduler, Epochs:", epochs, "Batch-size:", batch_size)
    print("==============epochs:", epochs)
    learning_rate = constant_learning_rate
    lr_schedule = sparse_soft_trees.ConstantLearningRate(
        learning_rate
    )
    optim = tf.keras.optimizers.SGD(lr_schedule)

    ### Soft Decision Tree parameters 
    num_trees = config['num_trees']
    depth = config['depth'] 
    # We only use a single layer of Tree Ensemble [Note layer is different than depth]. 
    # kernel_l2 = trial.suggest_loguniform('kernel_l2', 1e-5, 1e-0)

    activation = tf.keras.activations.sigmoid

    use_annealing = config['use_annealing']
    kernel_l2 = config['kernel_l2']
    kernel_l2 = kernel_l2/(num_trees*(2**depth - 1))
    kernel_regularizer = tf.keras.regularizers.L2(kernel_l2)
    if use_annealing:
        kernel_constraint = config['kernel_constraint']/num_features
        temperature = config['temperature']
        kernel_constraint=sparse_soft_trees.ProximalGroupL0(lr=lr_schedule, lam=kernel_constraint, temperature=temperature, use_annealing=True, name='ProximalGroupL0')
    else:
        kernel_constraint = config['kernel_constraint']/num_features
        kernel_constraint=sparse_soft_trees.ProximalGroupL0(lr=lr_schedule, lam=kernel_constraint, use_annealing=False, name='ProximalGroupL0')
    print("===========kernel_regularizer:", kernel_regularizer)
    print("===========kernel_constraint:", kernel_constraint)
            
    ### Loss parameters
    loss_criteria = config['loss_criteria']
    # exponentials and sigmoids inside distributions
    output_activation = 'linear'
    loss = losses.NegativeLogLikelihood()
        
    ### Optimization parameters
    if loss_criteria in ['mse']:
        pass
    elif loss_criteria=='cross-entropy':
        leaf_dims = (num_classes, )
        x = tf.keras.layers.Input(name='input', shape=data_processed.x_train_processed.shape[1:])
        submodel = models.create_model(
            x,
            num_trees,
            depth,
            leaf_dims,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
        )
        x = submodel.input
        outputs = submodel(x)
        # print(outputs)
        ypred = tf.keras.layers.Activation('linear')(outputs)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model = tf.keras.Model(inputs=x, outputs=ypred)
                
    model.summary()

#     model.summary()
    # model.layers[1].summary()

        
    if data_type=='regression':
        monitor = 'val_loss'
        metrics = []
    elif data_type=='classification':
        monitor = 'val_accuracy'
        metrics = ['accuracy']
    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    cb = sparse_soft_trees.SparsityHistory()
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        cb
    ]    
    if early_stopping:
        callbacks.append(
            sparse_soft_trees.EarlyStopping(
                config['cardinality'], monitor=monitor, patience=100, verbose=1, mode='auto', restore_best_weights=True
            ),
        )
    # print("====================y.shape", data_processed.y_train_processed.shape)
    if len(get_available_gpus())==0:
        history = model.fit(x=data_processed.x_train_processed, 
                  y=data_processed.y_train_processed,
                  sample_weight=data_processed.w_train,
                  epochs=epochs, 
                  batch_size=batch_size, 
                  shuffle=True,
                  callbacks=callbacks,
                  validation_data=(data_processed.x_valid_processed, data_processed.y_valid_processed, data_processed.w_valid),
                  verbose=1, 
                  )  
    else:
        with tf.device(get_available_gpus()[0]):
            history = model.fit(x=data_processed.x_train_processed, 
                      y=data_processed.y_train_processed,
                      sample_weight=data_processed.w_train,
                      epochs=epochs, 
                      batch_size=batch_size, 
                      shuffle=True,
                      callbacks=callbacks,
                      validation_data=(data_processed.x_valid_processed, data_processed.y_valid_processed, data_processed.w_valid),
                      verbose=0, 
                      )  
    number_of_epochs_it_ran = len(history.history['loss'])
    
    val_loss_history = history.history['val_loss']

    if early_stopping:
        best_epoch = callbacks[-1].best_epoch # np.argmin(val_loss_history) + 1
    else:
        best_epoch = len(val_loss_history)
    print("============best_epoch:", best_epoch)
    print("============number_of_epochs_it_ran:", number_of_epochs_it_ran)

    feature_sparsity_history = cb.selected_features[1:(best_epoch+1)]
    approximate_feature_sparsity_history = cb.approximately_selected_features[1:(best_epoch+1)]
    weight_sparsity_history = cb.selected_weights[1:(best_epoch+1)]
    approximate_weight_sparsity_history = cb.approximately_selected_weights[1:(best_epoch+1)]
    print('=================feature_sparsity:', feature_sparsity_history[-1])
    print('=================feature_approx_sparsity:', approximate_feature_sparsity_history[-1])
    print('=================weight_sparsity:', weight_sparsity_history[-1])
    print('=================weight_approx_sparsity:', approximate_weight_sparsity_history[-1])
        
    with tf.device(get_available_cpus()[0]):
        # Check for infinite loss
        training_loss = model.evaluate(data_processed.x_train_processed,
                                       data_processed.y_train_processed,
                                       sample_weight=data_processed.w_train,
                                       batch_size=batch_size,
                                       verbose=0)

        if np.isfinite(np.sum(training_loss)) or ~np.isnan(np.sum(training_loss)):
            # Evaluation
            
            if loss_criteria in ['mse']:
                pass
            elif loss_criteria in ['cross-entropy']:                
                y_valid_pred = model.predict(data_processed.x_valid_processed)
                y_valid_prob = tf.nn.softmax(y_valid_pred, axis=1).numpy()
                y_valid_pred_classes = np.argmax(y_valid_prob, axis=1)
                classes = np.shape(y_valid_prob)[1]
                if classes==2:
                    y_valid_prob = tf.gather(y_valid_prob, indices=[1], axis=1).numpy()
                accuracy_valid = accuracy_score(data_processed.y_valid_processed, y_valid_pred_classes)
                auc_valid = roc_auc_score(
                    data_processed.y_valid_processed,
                    y_valid_prob,
                    multi_class='ovo'
                )
                
                y_test_pred = model.predict(data_processed.x_test_processed)
                y_test_prob = tf.nn.softmax(y_test_pred, axis=1).numpy()
                y_test_pred_classes = np.argmax(y_test_prob, axis=1)
                if classes==2:
                    y_test_prob = tf.gather(y_test_prob, indices=[1], axis=1).numpy()
                accuracy_test = accuracy_score(data_processed.y_test_processed, y_test_pred_classes)
                auc_test = roc_auc_score(
                    data_processed.y_test_processed,
                    y_test_prob,
                    multi_class='ovo'
                )
                print('accuracy (valid):', accuracy_valid)
                print('accuracy (test):', accuracy_test)
                print('auc (valid):', auc_valid)
                print('auc (test):', auc_test)
#                 if feature_sparsity_history[-1]>(0.5*data_processed.x_train_processed.shape[1]):
#                     accuracy_valid = np.nan
#                     accuracy_test = np.nan
#                     auc_valid = np.nan
#                     auc_test = np.nan
        else:
            if loss_criteria in ['mse']: 
                pass
            elif loss_criteria in ['cross-entropy']:
                accuracy_valid = np.nan
                accuracy_test = np.nan
                auc_valid = np.nan
                auc_test = np.nan
    
    if loss_criteria in ['mse']:   
        valid_criteria_multitask = mse_valid
    elif loss_criteria in ['cross-entropy']:
        valid_criteria_multitask = auc_valid
    else:
        raise ValueError("loss criteria {} is not supported".format(loss_criteria))
        
    print('Valid Criteria:', valid_criteria_multitask)
    
    # Save trained model to a file.
    trial_path = os.path.join(path, "trials", "trial{}".format(trial.number))
    # model.save(os.path.join(trial_path, "model"))

    if loss_criteria in ['mse']:
        pass
    elif loss_criteria in ['cross-entropy']:
        trial.set_user_attr("accuracy_valid", accuracy_valid)
        trial.set_user_attr("accuracy_test", accuracy_test)
        trial.set_user_attr("auc_valid", auc_valid)
        trial.set_user_attr("auc_test", auc_test)
    trial.set_user_attr("num_epochs", number_of_epochs_it_ran)
    trial.set_user_attr("val_loss_history", val_loss_history)
    trial.set_user_attr("feature_sparsity_history", feature_sparsity_history)    
    trial.set_user_attr("approximate_feature_sparsity_history", approximate_feature_sparsity_history)    
    trial.set_user_attr("weight_sparsity_history", weight_sparsity_history)    
    trial.set_user_attr("approximate_weight_sparsity_history", approximate_weight_sparsity_history)    
    trial.set_user_attr("feature_sparsity", feature_sparsity_history[-1])    
    trial.set_user_attr("approximate_feature_sparsity", approximate_feature_sparsity_history[-1])    
    trial.set_user_attr("weight_sparsity", weight_sparsity_history[-1])    
    trial.set_user_attr("approximate_weight_sparsity", approximate_weight_sparsity_history[-1])    
    
    return valid_criteria_multitask



def main():
    """
    Parses the hyperparameter search command, defines the hyperparameter space, instantiates each model,
    and launches a hyperparameter search (by calling multiple times the train loop defined in train_tasks.py).
    inputs: None
    outputs: None.
    """
    # We define the parser.

    parser = argparse.ArgumentParser(description='Skinny Trees for feature selection.')

    # Data Arguments
    parser.add_argument('--load_directory', dest='load_directory',  type=str, default='s3://cortex-mit1003-lmdl-workbucket/public-datasets-processed')
    parser.add_argument('--data', dest='data',  type=str, default='churn')
    parser.add_argument('--data_type', dest='data_type',  type=str, default='classification')
    parser.add_argument('--seed', dest='seed',  type=int, default=8)
    parser.add_argument('--val_size', dest='val_size',  type=float, default=0.2)
    parser.add_argument('--test_size', dest='test_size',  type=float, default=0.2)

    # Model Arguments
    parser.add_argument('--anneal', action='store_true') # for dense-to-sparse scheduler

    parser.add_argument('--batch_size_scaler', dest='batch_size_scaler',  type=int)
    parser.add_argument('--constant_learning_rate', dest='constant_learning_rate',  type=float)
    parser.add_argument('--num_trees', dest='num_trees',  type=int)
    parser.add_argument('--depth', dest='depth',  type=int)
    parser.add_argument('--epochs', dest='epochs',  type=int)
    parser.add_argument('--kernel_constraint', dest='kernel_constraint',  type=float) # for Group L0-L2
    parser.add_argument('--kernel_l2', dest='kernel_l2',  type=float) # for Group L0-L2
    parser.add_argument('--kernel_regularizer', dest='kernel_regularizer',  type=float) # for Group Lasso
    parser.add_argument('--temperature', dest='temperature',  type=float) # for dense-to-sparse
    parser.add_argument('--use_passed_hyperparameters',action='store_true')

    
    
    # Algorithm Arguments
    parser.add_argument('--loss', dest='loss',  type=str, default='cross-entropy')
    parser.add_argument('--max_trees', dest='max_trees',  type=int, default=100)
    parser.add_argument('--max_depth', dest='max_depth',  type=int, default=4)
    parser.add_argument('--max_epochs', dest='max_epochs',  type=int, default=200)
    parser.add_argument('--n_trials', dest='n_trials',  type=int, default=2)
    parser.add_argument('--patience', dest='patience',  type=int, default=25)

    # Logging Arguments
    parser.add_argument('--version', dest='version',  type=int, default=1)
    parser.add_argument('--save_directory', dest='save_directory',  type=str, default='./logs/soft_trees/publicdata')

    # Tuning Arguments
    parser.add_argument('--tuning_criteria', dest='tuning_criteria',  type=str, default='auc')
    parser.add_argument('--tuning_seed', dest='tuning_seed',  type=int, default=1)

    args = parser.parse_args()

    print("Load Directory:", args.load_directory)
    print("Data:", args.data)
    data_processed = data_utils.load_processed_classification_public_data(
        args.data,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        path=args.load_directory,
    )


    path = os.path.join(args.save_directory, args.data, args.loss, "{}.{}".format(args.version, args.tuning_seed))
    os.makedirs(path, exist_ok=True)

    config = dict({
        'early_stopping': False,
        'loss_criteria': args.loss,
    })
    
    if args.use_passed_hyperparameters:
        config.update({
            'constant_batch_size': 16, # trial.suggest_categorical('constant_batch_size', [16])
            'batch_size_scaler': args.batch_size_scaler,
            'constant_learning_rate': args.constant_learning_rate,
            'num_trees': args.num_trees,
            'depth': args.depth,
            'use_annealing': args.anneal, # trial.suggest_categorical('use_annealing', [args.anneal])
            'epochs': args.epochs,
            'kernel_constraint': args.kernel_constraint,
            'kernel_l2': args.kernel_l2,
            'kernel_regularizer': args.kernel_regularizer,
            'temperature': args.temperature
        }) 
    else:
        pass
    
    
    
    start = timeit.default_timer()

    global trial_nb
    trial_nb = 0

    filename = "study"
    filename = filename + "-{}".format(args.architecture)
    if args.anneal:
        filename = filename + "-anneal"
    filename = filename + "-seed{}".format(args.seed)
    global filepath
    filepath = os.path.join(path, filename+".csv")
    global studypath
    studypath = os.path.join(path, filename+".pkl")
    
    global n_trials
    n_trials = args.n_trials
    
    global direction
    if args.loss in ['mse']:
        direction = 'minimize'
    elif args.loss in ['cross-entropy']:
        direction = 'maximize'
    
    
    study = optuna.create_study(sampler=RandomSampler(seed=args.tuning_seed), direction=direction)
    objective_wrapper = lambda trial: objective(
        trial,
        data_processed,
        args.data_type,
        path,
        args,
        config,
        args.use_passed_hyperparameters
    )
    study.optimize(objective_wrapper, n_trials=args.n_trials, callbacks=[logging_callback])

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
    df_study.to_csv(filepath)
    joblib.dump(study, studypath) 

def logging_callback(study, frozen_trial, save_freq=5):
    """
    Callback function executed at the end of each trial to save information on the disk.
    inputs:
        - optuna study which we enrich with our trials
        - the previous trial
        - the frequency at which we save our information on the disk
    outputs:
        - None
    """
#     # We print in our logs the best val quantity observed so far, just to keep track.
#     if "previous_best_value" not in study.user_attrs.keys():
#         study.set_user_attr("previous_best_value", frozen_trial.values[0])
#         print(" => Best:", frozen_trial.values[0])
#     else:
#         previous_best_value = study.user_attrs["previous_best_value"]
#         if direction=='minimize':
#             if previous_best_value > frozen_trial.values[0]:
#                 study.set_user_attr("previous_best_value", frozen_trial.values[0])
#                 print(" => Best:", frozen_trial.values[0])
#             else:
#                 print(" => Best:", previous_best_value)
#         else:
#             if previous_best_value < frozen_trial.values[0]:
#                 study.set_user_attr("previous_best_value", frozen_trial.values[0])
#                 print(" => Best:", frozen_trial.values[0])
#             else:
#                 print(" => Best:", previous_best_value)
            
    
    # We save a csv of the study every 5 trials 
    # AND also a Pareto front graph of our val loss-val sparsity hyperparam search 
    # (plus some hyperparameter importance graphs for each of the 2 objectives).
    if trial_nb % save_freq or trial_nb >= n_trials-1:
        # saving of the csv of the study
        joblib.dump(study, studypath)  
        df = study.trials_dataframe()
        df.to_csv(filepath)

if __name__ == "__main__":
    main()
