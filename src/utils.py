import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from typing import Callable

class ConstantLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses constant learning rate.
    The schedule is a 1-arg callable that produces a learning rate when passed the current optimizer step.
    This can be useful for changing the learning rate value across different invocations of optimizer functions.
    It is computed as:
    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer` as the learning rate.
    ```python
    ...
    learning_rate = 0.1
    learning_rate_fn = tf.keras.optimizers.schedules.ConstantLearningRate(
        learning_rate,
        )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate_fn),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        )
    model.fit(data, labels, epochs=5)
    ```
    The learning rate schedule is also serializable and deserializable using
    `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.
    Returns:
        A 1-arg callable learning rate schedule that takes the current optimizer
        step and outputs the learning rate, a scalar `Tensor` of the same
        type as `learning_rate`.
    """

    def __init__(
        self,
        learning_rate,
        name=None):
        """Applies a polynomial decay to the learning rate.
        Args:
        learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The learning rate.
        name: String.  Optional name of the operation. Defaults to
            'Constant'.
        """
        super(ConstantLearningRate, self).__init__()
        self.learning_rate = learning_rate
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "ConstantLearningRate") as name:
            learning_rate = tf.convert_to_tensor(self.learning_rate, name="low_learning_rate")
            dtype = learning_rate.dtype
            global_step = tf.cast(step, dtype)
            return learning_rate

    def get_config(self):
        return {
            "learning_rate": self.learning_rate,
            "name": self.name
        }

class LinearEpochGradualWarmupPolynomialDecayLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses a Linear Epoch Gradual Warmup and Polynomial decay schedule.
    It is commonly observed that a linear ramp-up and monotonically decreasing learning rate, whose degree of change
    is carefully chosen, results in a better performing model. This schedule applies a polynomial decay function to an
    optimizer step, given a provided `low_learning_rate`, to reach an `peal_learning_rate` in the given `warmup_steps`,
    and reach a low_learning rate in the remaining steps via a polynomial decay.
    It requires a `step` value to compute the learning rate. You can just pass a TensorFlow variable that you
    increment at each training step.
    The schedule is a 1-arg callable that produces a decayed learning rate when passed the current optimizer step.
    This can be useful for changing the learning rate value across different invocations of optimizer functions.
    It is computed as:
    ```python
    def decayed_learning_rate(step):
        step = min(step, decay_steps)
        return ((low_learning_rate - peak_learning_rate) *
            (1 - step / decay_steps) ^ (power)
           ) + end_learning_rate
    ```
    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer` as the learning rate.
    Example: Fit a model while ramping up from 0.01 to 0.1 in 1000 steps and decaying from 0.1 to 0.01 in 9000
        steps using sqrt (i.e. power=2.0):
    ```python
    ...
    peak_learning_rate = 0.1
    low_learning_rate = 0.01
    total_steps = 1000
    total_steps = 10000
    learning_rate_fn = LinearEpochGradualWarmupPolynomialDecayLearningRate(
        low_learning_rate,
        peak_learning_rate,
        warmup_steps,
        total_steps,
        power=2.0)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate_fn),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        )
    model.fit(data, labels, epochs=5)
    ```
    The learning rate schedule is also serializable and deserializable using
    `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.
    Returns:
        A 1-arg callable learning rate schedule that takes the current optimizer step and outputs the learning rate, a scalar 
        `Tensor` of the same type as `low_learning_rate`.
    """

    def __init__(
        self,
        low_learning_rate,
        peak_learning_rate,
        warmup_steps,
        total_steps,
        power=2.0,
        name=None):
        """Applies a Linear Epoch Gradual Warmup and Polynomial decay to the learning rate.
        Args:
        low_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The initial learning rate.
        peak_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The peak learning rate.
        warmup_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Must be positive.  See the warmup computation above.
        total_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Must be positive.  See the decay computation above.
        power: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The power of the polynomial. Defaults to linear, 1.0.
        name: String.  Optional name of the operation. Defaults to
            'PolynomialDecay'.
        """
        super(LinearEpochGradualWarmupPolynomialDecayLearningRate, self).__init__()
        self.low_learning_rate = low_learning_rate
        self.warmup_steps = warmup_steps
        self.peak_learning_rate = peak_learning_rate
        self.total_steps = total_steps
        self.power = power
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "LinearEpochGradualWarmupPolynomialDecayLearningRate") as name:
            low_learning_rate = tf.convert_to_tensor(self.low_learning_rate, name="low_learning_rate")
            dtype = low_learning_rate.dtype
            peak_learning_rate = tf.cast(self.peak_learning_rate, dtype)
            warmup_steps = tf.cast(self.warmup_steps, dtype)
            power = tf.cast(self.power, dtype)

            global_step = tf.cast(step, dtype)
            warmup_percent_done = global_step / warmup_steps
            warmup_learning_rate = (peak_learning_rate - low_learning_rate) * tf.math.pow(
                warmup_percent_done, tf.cast(1.0, dtype)
            ) + low_learning_rate

            total_steps = tf.cast(self.total_steps, dtype)
            decay_steps = total_steps - warmup_steps
            p = tf.divide(
                tf.minimum(global_step - warmup_steps, decay_steps),
                decay_steps
            )
            decay_learning_rate = tf.add(
                tf.multiply(
                    peak_learning_rate - low_learning_rate,
                    tf.pow(1 - p, power)
                ),
                low_learning_rate,
                name="decay_learning_rate"
            )

            learning_rate = tf.cond(
                global_step < warmup_steps,
                lambda: warmup_learning_rate,
                lambda: decay_learning_rate,
                name="learning_rate",
            )
            return learning_rate

    def get_config(self):
        return {
            "low_learning_rate": self.low_learning_rate,
            "peak_learning_rate": self.peak_learning_rate,
            "warm_steps": self.warm_steps,
            "total_steps": self.total_steps,
            "power": self.power,
            "name": self.name
        }

class EarlyStopping(tf.keras.callbacks.Callback):
    """Stop training when a monitored metric has stopped improving.

    Assuming the goal of a training is to minimize the loss. With this, the
    metric to be monitored would be `'loss'`, and mode would be `'min'`. A
    `model.fit()` training loop will check at end of every epoch whether
    the loss is no longer decreasing, considering the `min_delta` and
    `patience` if applicable. Once it's found no longer decreasing,
    `model.stop_training` is marked True and the training terminates.

    The quantity to be monitored needs to be available in `logs` dict.
    To make it so, pass the loss or metrics at `model.compile()`.

    Args:
      monitor: Quantity to be monitored.
      min_delta: Minimum change in the monitored quantity
          to qualify as an improvement, i.e. an absolute
          change of less than min_delta, will count as no
          improvement.
      patience: Number of epochs with no improvement
          after which training will be stopped.
      verbose: Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1
          displays messages when the callback takes an action.
      mode: One of `{"auto", "min", "max"}`. In `min` mode,
          training will stop when the quantity
          monitored has stopped decreasing; in `"max"`
          mode it will stop when the quantity
          monitored has stopped increasing; in `"auto"`
          mode, the direction is automatically inferred
          from the name of the monitored quantity.
      baseline: Baseline value for the monitored quantity.
          Training will stop if the model doesn't show improvement over the
          baseline.
      restore_best_weights: Whether to restore model weights from
          the epoch with the best value of the monitored quantity.
          If False, the model weights obtained at the last step of
          training are used. An epoch will be restored regardless
          of the performance relative to the `baseline`. If no epoch
          improves on `baseline`, training will run for `patience`
          epochs and restore weights from the best epoch in that set.
      start_from_epoch: Number of epochs to wait before starting
          to monitor improvement. This allows for a warm-up period in which
          no improvement is expected and thus training will not be stopped.


    Example:

    >>> callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    >>> # This callback will stop the training when there is no improvement in
    >>> # the loss for three consecutive epochs.
    >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
    >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
    ...                     epochs=10, batch_size=1, callbacks=[callback],
    ...                     verbose=0)
    >>> len(history.history['loss'])  # Only 4 epochs are run.
    4
    """

    def __init__(
        self,
        cardinality,
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0,
    ):
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.start_from_epoch = start_from_epoch
        self.cardinality = cardinality

        if mode not in ["auto", "min", "max"]:
            logging.warning(
                "EarlyStopping mode %s is unknown, fallback to auto mode.",
                mode,
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            if (
                self.monitor.endswith("acc")
                or self.monitor.endswith("accuracy")
                or self.monitor.endswith("auc")
            ):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None or epoch < self.start_from_epoch:
            # If no monitor value exists or still in initial warm-up stage.
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = self.model.get_weights()
            self.num_selected_features = np.sum(
                np.linalg.norm(
                    self.model.layers[1].layers[1].dense_layer.get_weights()[0],
                    axis=1
                )>0.0
            )

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
                self.num_selected_features = np.sum(
                    np.linalg.norm(
                        self.model.layers[1].layers[1].dense_layer.get_weights()[0],
                        axis=1
                    )>0.0
                )
            # Only restart wait if we beat both the baseline and our previous
            # best.
            if self.baseline is None or self._is_improvement(
                current, self.baseline
            ):
                self.wait = 0
            return

        # Only check after the first epoch.
        if self.wait >= self.patience and epoch > 0 and tf.math.less_equal(self.num_selected_features, self.cardinality):
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    tf.print(
                        "Restoring model weights from "
                        "the end of the best epoch: "
                        f"{self.best_epoch + 1}."
                    )
                    
                # best weights from earlier epochs may not satisfy cardinality. 
                # Therefore, check cardinality of current weights and best weights.
                # if best weights satisfy cardinality, then restore them
                # otherwise restore current weights
                current_weights = self.model.get_weights()
                current_num_selected_features = np.sum(
                    np.linalg.norm(
                        self.model.layers[1].layers[1].dense_layer.get_weights()[0],
                        axis=1
                    )>0.0
                )
                self.model.set_weights(self.best_weights)
                best_num_selected_features = np.sum(
                    np.linalg.norm(
                        self.model.layers[1].layers[1].dense_layer.get_weights()[0],
                        axis=1
                    )>0.0
                )
                
                if tf.math.less_equal(best_num_selected_features, self.cardinality):
                    self.model.set_weights(self.best_weights)
                    tf.print("Restoring best weights, which satisfy cardinality")
                else:
                    self.model.set_weights(current_weights)
                    self.best_epoch = epoch
                    tf.print("Restoring current weights, which may or may not satisfy cardinality")

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            tf.print(
                f"Epoch {self.stopped_epoch + 1}: early stopping"
            )

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(
                "Early stopping conditioned on cardinality `%s` and metric `%s`"
                "which is not available. Available metrics are: %s",
                self.cardinality,
                self.monitor,
                ",".join(list(logs.keys())),
            )
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)
