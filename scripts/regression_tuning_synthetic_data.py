import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib

import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import sys
import pandas as pd, numpy as np,  matplotlib.pyplot as plt
import optuna
from optuna.samplers import RandomSampler
import timeit
import joblib
from copy import deepcopy
from sklearn.metrics import mean_squared_error, recall_score, f1_score, accuracy_score, roc_auc_score

import pathlib
sys.path.insert(0, os.path.abspath(str(pathlib.Path(__file__).absolute()).split('src')[0]))
from src import utils_multitask

from src import models
from src import sparse_soft_trees


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']


def objective(
    trial,
    data_processed,
    data_type,
    path,
    args,
    config,
    ):
    
    """ Clear the backend (TensorFlow). See:
    https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session
    """
    K.clear_session() 

    print("====================path", path)
    num_features = data_processed.x_train_processed.shape[1]

    config.update({
#         'constant_batch_size': trial.suggest_categorical('constant_batch_size', [64]),
        'constant_batch_size': trial.suggest_categorical('constant_batch_size', [16]),
        'batch_size_scaler': trial.suggest_categorical('batch_size_scaler', np.arange(1,16)),
        'constant_learning_rate': trial.suggest_loguniform('constant_learning_rate', 1e-3, 1e-1),
        'num_trees': trial.suggest_int('num_trees', 1, args.max_trees),
#         'num_trees': trial.suggest_int('num_trees', args.max_trees, args.max_trees),
        'depth': trial.suggest_int('depth', 1, args.max_depth),
#         'depth': trial.suggest_int('depth', args.max_depth, args.max_depth),
        'use_annealing': trial.suggest_categorical('use_annealing', [args.anneal])
    })

    if config['early_stopping']:
        config['epochs'] = args.max_epochs
    else:
        config['epochs'] = trial.suggest_int('epochs', 5, args.max_epochs, 5) # [5,500] originally

    config['kernel_l2'] = trial.suggest_loguniform('kernel_l2', 1e-3, 1e0)
    if config['use_annealing']:
        config['kernel_constraint'] = trial.suggest_categorical('kernel_constraint', [100.0])
        config['temperature'] = trial.suggest_loguniform('temperature', 1e-4, 1e-1)
    else:
        config['kernel_constraint'] = trial.suggest_loguniform('kernel_constraint', 1e+0, 1e+4)
            
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
        
#    epochs = trial.suggest_int('epochs_new', 5, epochs)
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
    num_layers = 1
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
        
    ### Optimization parameters
    if loss_criteria in ['mse']:
        leaf_dims = (1, )
        x = tf.keras.layers.Input(name='input', shape=data_processed.x_train_processed.shape[1:])
        submodel = models_multitask.create_multitask_sparse_submodel(
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
        loss = tf.keras.losses.MeanSquaredError()
        model = tf.keras.Model(inputs=x, outputs=ypred)
    elif loss_criteria=='cross-entropy':
        leaf_dims = (num_classes, )
        x = tf.keras.layers.Input(name='input', shape=data_processed.x_train_processed.shape[1:])
        submodel = models_multitask.create_multitask_sparse_submodel(
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
    # model.layers[1].summary()

        
    if data_type=='regression':
        monitor = 'val_mse'
        metrics = ['mse']
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
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor, patience=50, verbose=1, mode='auto', restore_best_weights=True
            ),
        )
    # print("====================y.shape", data_processed.y_train_processed.shape)
    if len(get_available_gpus())==0:
        history = model.fit(x=data_processed.x_train_processed, 
                  y=data_processed.y_train_processed,
                  epochs=epochs, 
                  batch_size=batch_size, 
                  shuffle=True,
                  callbacks=callbacks,
                  validation_data=(data_processed.x_valid_processed, data_processed.y_valid_processed),
                  verbose=0, 
                  )  
    else:
        with tf.device(get_available_gpus()[0]):
            history = model.fit(x=data_processed.x_train_processed, 
                      y=data_processed.y_train_processed,
                      epochs=epochs, 
                      batch_size=batch_size, 
                      shuffle=True,
                      callbacks=callbacks,
                      validation_data=(data_processed.x_valid_processed, data_processed.y_valid_processed),
                      verbose=0, 
                      )  
    number_of_epochs_it_ran = len(history.history['loss'])
    
    val_loss_history = history.history['val_loss']

    if early_stopping:
        best_epoch = np.argmin(val_loss_history) + 1
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
                                       batch_size=batch_size,
                                       verbose=0)

        weights = model.layers[1].layers[1].dense_layer.get_weights()[0]
        if np.isfinite(np.sum(training_loss)) or ~np.isnan(np.sum(training_loss)):
            # Evaluation
            feature_support = np.linalg.norm(weights, axis=1)>0.0
            
            if loss_criteria in ['mse']:
                y_valid_pred = model.predict(data_processed.x_valid_processed, verbose=0)
                y_test_pred = model.predict(data_processed.x_test_processed, verbose=0)
                mse_valid = mean_squared_error(data_processed.y_valid_processed, y_valid_pred)
                mse_test = mean_squared_error(data_processed.y_test_processed, y_test_pred)
                print('mse (valid):', mse_valid)
                print('mse (test):', mse_test)
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
                
            
        else:
            feature_support = np.ones(weights.shape[0])
            
            if loss_criteria in ['mse']: 
                mse_valid = np.inf
                mse_test = np.inf
            elif loss_criteria in ['cross-entropy']:
                accuracy_valid = 0.0
                accuracy_test = 0.0
                auc_valid = 0.0
                auc_test = 0.0
    
    if loss_criteria in ['mse']:   
        valid_criteria_multitask = np.sum(mse_valid)
    elif loss_criteria in ['cross-entropy']:
        valid_criteria_multitask = auc_valid
    else:
        raise ValueError("loss criteria {} is not supported".format(loss_criteria))
        
    print('Valid Criteria:', valid_criteria_multitask)

    
    # Compute FPR and FNR for features
    tpr = recall_score(data_processed.feature_support_truth, feature_support)   # it is better to name it y_test 
    # to calculate, tnr we need to set the positive label to the other class
    # I assume your negative class consists of 0, if it is -1, change 0 below to that value
    tnr = recall_score(data_processed.feature_support_truth, feature_support, pos_label=0) 
    fpr = 1 - tnr
    fnr = 1 - tpr   
    f1 = f1_score(data_processed.feature_support_truth, feature_support)    
    
    
    
    # Save trained model to a file.
    trial_path = os.path.join(path, "trials", "trial{}".format(trial.number))
    # model.save(os.path.join(trial_path, "model"))

    if loss_criteria in ['mse']:
        trial.set_user_attr("mse_valid", mse_valid)
        trial.set_user_attr("mse_test", mse_test)
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
    trial.set_user_attr("feature_support_truth", data_processed.feature_support_truth)        
    trial.set_user_attr("feature_support", feature_support)        
    trial.set_user_attr("tpr", tpr)    
    trial.set_user_attr("tnr", tnr)    
    trial.set_user_attr("fpr", fpr)    
    trial.set_user_attr("fnr", fnr)    
    trial.set_user_attr("f1", f1)    
    
    return valid_criteria_multitask
