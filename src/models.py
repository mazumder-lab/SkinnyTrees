"""Creates models for multitask learning with Neural Networks.
"""
import tensorflow as tf
from tensorflow.keras import models

import pathlib
import os
import sys
sys.path.insert(0, os.path.abspath(str(pathlib.Path(__file__).absolute()).split('src')[0]))
from src import sparse_soft_trees

def create_model(
    x,
    num_trees,
    depth,
    leaf_dims,
    activation='sigmoid',
    kernel_regularizer=tf.keras.regularizers.L2(0.0),
    kernel_constraint=None,
    ):
    """Creates a submodel for a task with soft decision trees.
    
    Args:
      x: Keras Input instance.
      num_trees: Number of trees in the ensemble, int scalar.
      depth: Depth of each tree. Note: in the current implementation,
        all trees are fully grown to depth, int scalar.
      leaf_dims: list of dimensions of leaf outputs,
        int tuple of shape (num_layers, ).
      activation: 'sigmoid'
      
    Returns:
      Keras submodel instantiation
    """
    
    y = sparse_soft_trees.TreeEnsembleWithGroupSparsity(
        num_trees,
        depth,
        leaf_dims[-1],
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_constraint=kernel_constraint
    )(x) 
        
    submodel = models.Model(inputs=x, outputs=y)
    return submodel