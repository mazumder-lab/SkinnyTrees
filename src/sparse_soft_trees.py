import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras import activations
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import backend as K
from typing import Callable

import pathlib
import os
import sys
sys.path.insert(0, os.path.abspath(str(pathlib.Path(__file__).absolute()).split('src')[0]))
from src import utils

class ProximalGroupL0(tf.keras.constraints.Constraint):
    """Applies group-wise proximal operator after gradient update for Group L0.
    
    Solves the following minimization problem:
        min_Z (1/(2*lr))*||Z-W_{t}))||_2^2 + lam_lo * sum_{j=1}^p ||Z_j||_0
        s.t. W_{t} = W_{t-1}-lr*Df(W_{t-1})
    
    Proximal Operator:
    For each group w_t, the proximal operator is a soft-thresholding operator:
        H_{lr*lam_l0, lr*lam_l2}(w_t) = {w_t  if ||w_t||_2 >= sqrt(2*lr*lam_l0),
                                        {0    o.w. 

    References:
        - End-to-end Feature Selection Approach for Learning Skinny Trees
          [https://arxiv.org/pdf/2310.18542.pdf]
        - Grouped Variable Selection with Discrete Optimization: Computational and Statistical Perspectives
          [https://arxiv.org/pdf/2104.07084.pdf]
        

    Inputs:
        w: Float Tensor of shape (num_features, num_trees*num_nodes).
        
    Returns:
        w: Float Tensor of shape (num_features, num_trees*num_nodes).
    """
    def __init__(self, lr=utils.ConstantLearningRate(0.), lam=0., use_annealing=False, temperature=0.1, name='ProximalGroupL0', **kwargs):
        super(ProximalGroupL0, self).__init__(**kwargs)
        self.lam = lam
        assert isinstance(lr, Callable)
        self.lr = lr
        self.use_annealing = use_annealing
        self.name=name
        self.iterations = tf.Variable(initial_value=0, trainable=False, name='iterations')
        if self.use_annealing:
            self.temperature = tf.constant(temperature)
    
    def __call__(self, w):
        self.iterations.assign_add(1)
        lam_lr = self.lam * self.lr(self.iterations)
        if self.use_annealing:
            scheduler = (1.0-tf.math.exp(
                -tf.cast(self.temperature, w.dtype)*tf.cast(self.iterations, dtype=w.dtype)
            ))
        else:
            scheduler = 1.0
        # The proximity operator for the group l0 is hard thresholding on each group.
        w_norm = tf.norm(w, ord='euclidean', axis=1)
        w_norm = tf.expand_dims(w_norm, axis=1)
        hard_threshold = tf.math.sqrt(
            2.0 * tf.cast(lam_lr * scheduler, dtype=w.dtype)
        )
        w = tf.where(
            tf.greater(
                w_norm,
                hard_threshold*tf.ones_like(w_norm)
            ),
            w,
            tf.zeros_like(w)
        )
        return w
        
    def get_config(self):
        config = super(ProximalGroupL0, self).get_config()
        config.update({"lam": self.lam})
        config.update({"use_annealing": self.use_annealing})    
        config.update({"temperature": self.temperature})    
        config.update({"iterations": self.iterations.numpy()})
        config.update({"lr": tf.keras.optimizers.schedules.serialize(self.lr)})
        return config


# Utility function to count active features in a Tree Ensemble
def count_selected_weights(model):    
    weights = model.layers[1].layers[1].dense_layer.get_weights()[0]
    return np.sum(np.mean(np.abs(weights)>0.0, axis=1))

def count_approximately_selected_weights(model):    
    weights = model.layers[1].layers[1].dense_layer.get_weights()[0]
    return np.sum(np.mean(np.abs(weights)>1e-4, axis=1))

def count_selected_features(model):    
    weights = model.layers[1].layers[1].dense_layer.get_weights()[0]
    return np.sum(np.linalg.norm(weights, axis=1)>0.0)

def count_approximately_selected_features(model):   
    weights = model.layers[1].layers[1].dense_layer.get_weights()[0]    
    return np.sum((1/np.sqrt(weights.shape[1]))*np.linalg.norm(weights, axis=1)>1e-4)

# Callback class to save training loss and the number of features
class SparsityHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.selected_features = [count_selected_features(self.model)]
        self.approximately_selected_features = [count_approximately_selected_features(self.model)]
        self.selected_weights = [count_selected_weights(self.model)]
        self.approximately_selected_weights = [count_approximately_selected_weights(self.model)]

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.selected_features.append(count_selected_features(self.model))
        self.approximately_selected_features.append(count_approximately_selected_features(self.model))
        self.selected_weights.append(count_selected_weights(self.model))
        self.approximately_selected_weights.append(count_approximately_selected_weights(self.model))

class TreeEnsembleWithGroupSparsity(tf.keras.layers.Layer):
    """An ensemble of soft decision trees.
    
    The layer returns the sum of the decision trees in the ensemble.
    Each soft tree returns a vector, whose dimension is specified using
    the `leaf_dims' parameter.
    
    Implementation Notes:
        This is a fully vectorized implementation. It treats the ensemble
        as one "super" tree, where every node stores a dense layer with 
        num_trees units, each corresponding to the hyperplane of one tree.
    
    Input:
        An input tensor of shape = (batch_size, ...)

    Output:
        An output tensor of shape = (batch_size, leaf_dims)
    """

    def __init__(self,
                 num_trees,
                 max_depth,
                 leaf_dims,
                 activation='sigmoid',
                 node_index=0,
                 internal_eps=0,
                 kernel_regularizer=regularizers.l2(0.0),
                 kernel_constraint=None):
        super(TreeEnsembleWithGroupSparsity, self).__init__()
        self.max_depth = max_depth
        self.leaf_dims = leaf_dims
        self.num_trees = num_trees
        self.node_index = node_index
        self.leaf = node_index >= 2**max_depth - 1
        self.max_split_nodes = 2**max_depth - 1
        self.internal_eps = internal_eps
        assert (
            isinstance(kernel_regularizer, regularizers.l2) and isinstance(kernel_constraint, ProximalGroupL0)
        ) or (
            isinstance(kernel_regularizer, regularizers.l2) and kernel_constraint is None
        )
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.activation = tf.keras.activations.get(activation)
        if not self.leaf:
            if self.node_index == 0:
                self.dense_layer = tf.keras.layers.Dense(
                    self.num_trees*self.max_split_nodes,
                    kernel_regularizer=self.kernel_regularizer,
                    kernel_constraint=self.kernel_constraint,
                    activation=None,
                )
            masking = np.zeros((1, self.num_trees, self.max_split_nodes))
            masking[:,:,self.node_index] = masking[:,:,self.node_index] + 1
            self.masking = tf.constant(masking, dtype=self.dtype)
            self.left_child = TreeEnsembleWithGroupSparsity(
                self.num_trees,
                self.max_depth,
                self.leaf_dims,
                activation=activation,
                node_index=2*self.node_index+1,
            )
            self.right_child = TreeEnsembleWithGroupSparsity(
                self.num_trees,
                self.max_depth,
                self.leaf_dims,
                activation=activation,
                node_index=2*self.node_index+2,
            )

    def build(self, input_shape):
        if self.leaf:
            self.leaf_weight = self.add_weight(
                shape=[1, self.leaf_dims, self.num_trees],
                trainable=True,
                name="Node-"+str(self.node_index))
        
    def call(self, input, prob=1.0):
        if self.node_index==0:
            output = self.dense_layer(input)
            output = tf.reshape(output, shape=(tf.shape(output)[0], self.num_trees, self.max_split_nodes))
            if isinstance(self.kernel_constraint, ProximalGroupL0):
                # lam * sum_{j=1}^p ||Z_j||_0
                w = self.dense_layer.kernel
                w_norm = tf.norm(w, ord='euclidean', axis=1) 
                lam = tf.cast(self.dense_layer.kernel.constraint.lam, dtype=self.dtype)
                if self.dense_layer.kernel.constraint.use_annealing:
                     lam = lam * (1.0 - tf.math.exp(
                         - tf.cast(self.dense_layer.kernel.constraint.temperature, w.dtype) * tf.cast(self.dense_layer.kernel.constraint.iterations, dtype=w.dtype)
                     ))
                self.add_metric(lam, name='lam')
                nnz = tf.reduce_sum(
                    tf.cast(tf.greater(w_norm, tf.zeros_like(w_norm)), dtype=self.dtype)
                )
                self.add_metric(nnz, name='nnz')
                regularization = lam * nnz                   
                self.add_metric(regularization, name='group-l0-reg')
        else:
            output = input
            
        if not self.leaf:
            # shape = (batch_size, num_trees)
            current_prob = tf.keras.backend.clip(
                self.activation(
                    tf.reduce_sum(output*self.masking, axis=-1)
                ),
                self.internal_eps,
                1-self.internal_eps)
            return self.left_child(output, current_prob * prob) + self.right_child(output, (1 - current_prob) * prob)
        else:
            # self.leaf_weight's shape = (1, self.leaf_dims, num_trees)
            # prob's shape = (batch_size, num_trees)
            return tf.math.reduce_sum(tf.expand_dims(prob, axis=1) * self.leaf_weight, axis=2)
        
    def get_config(self):
        config = super(TreeEnsembleWithGroupSparsity, self).get_config()
        config.update({"num_trees": self.num_trees})
        config.update({"max_depth": self.max_depth})
        config.update({"leaf_dims": self.leaf_dims})
        config.update({"node_index": self.node_index})
        config.update({"activation": activations.serialize(self.activation)})
        config.update({"kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer)})
        return config