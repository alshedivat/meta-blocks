""""Optimization-related functionality.

Functions `compute_<optimizer_name>_updates` implement optimizer-specific
computation of the updates and return corresponding tensors. This is needed
for inner-loop optimization: subclasses of tf.train.Optimizer do not compute
updates in the symbolic form and instead implement specific update ops that
directly act on the provided variables.
"""

import functools

import tensorflow.compat.v1 as tf

from meta_blocks.optimizers.multistep_optimizer import MultistepAdamOptimizer

DEFAULT_OPTIMIZER = functools.partial(tf.train.AdamOptimizer, beta1=0)


def get(name=None, name_prefix=None, **kwargs):
    if name is None:
        kwarg_names = ["learning_rate", "beta2", "epsilon"]
        opt_kwargs = {key: kwargs[key] for key in kwarg_names if key in kwargs}
        opt_name = (name_prefix or "") + tf.train.AdamOptimizer.__name__
        return DEFAULT_OPTIMIZER(name=opt_name, **opt_kwargs)
    elif name == "adam":
        kwarg_names = ["learning_rate", "beta1", "beta2", "epsilon"]
        opt_kwargs = {key: kwargs[key] for key in kwarg_names if key in kwargs}
        opt_name = (name_prefix or "") + tf.train.AdamOptimizer.__name__
        return tf.train.AdamOptimizer(name=opt_name, **opt_kwargs)
    elif name == "multistep_adam":
        kwarg_names = ["learning_rate", "beta1", "beta2", "epsilon", "n"]
        opt_kwargs = {key: kwargs[key] for key in kwarg_names if key in kwargs}
        opt_name = (name_prefix or "") + MultistepAdamOptimizer.__name__
        return MultistepAdamOptimizer(name=opt_name, **opt_kwargs)
    elif name == "sgd":
        kwarg_names = ["learning_rate"]
        opt_kwargs = {key: kwargs[key] for key in kwarg_names if key in kwargs}
        name = (name_prefix or "") + tf.train.GradientDescentOptimizer.__name__
        return tf.train.GradientDescentOptimizer(name=name, **opt_kwargs)
    else:
        raise ValueError(f"Unknown optimizer {name}")


def compute_adam_updates(self, grads_and_vars):
    """Constructs and returns tensors of Adam-updated parameters."""
    grads_and_vars = tuple(grads_and_vars)
    var_list = [v for _, v in grads_and_vars]

    # Calculate Adam updates.
    with tf.name_scope(self._name):
        # Initialize.
        with tf.init_scope():
            self._create_slots(var_list)
        self._prepare()

        # Compute updates for each variable.
        var_updates = []
        for grad, var in grads_and_vars:
            if grad is None:
                var_updates.append(None)
                continue

            # Setup.
            beta1_power, beta2_power = self._get_beta_accumulators()
            beta1_power = tf.cast(beta1_power, var.dtype.base_dtype)
            beta2_power = tf.cast(beta2_power, var.dtype.base_dtype)
            lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
            beta1_t = tf.cast(self._beta1_t, var.dtype.base_dtype)
            beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
            epsilon_t = tf.cast(self._epsilon_t, var.dtype.base_dtype)
            lr = lr_t * tf.sqrt(1 - beta2_power) / (1 - beta1_power)

            # m_t = beta1 * m + (1 - beta1) * g_t.
            m = self.get_slot(var, "m")
            m_t = tf.assign(
                m, beta1_t * m + grad * (1 - beta1_t), use_locking=self._use_locking
            )

            # v_t = beta2 * v + (1 - beta2) * (g_t * g_t).
            v = self.get_slot(var, "v")
            v_t = tf.assign(
                v,
                beta2_t * v + (grad * grad) * (1 - beta2_t),
                use_locking=self._use_locking,
            )

            var_updates.append(lr * m_t / (tf.sqrt(v_t) + epsilon_t))

        # Update power accumulators.
        with tf.control_dependencies(var_updates):
            beta1_power, beta2_power = self._get_beta_accumulators()
            update_beta1 = tf.assign(
                beta1_power, beta1_power * self._beta1_t, use_locking=self._use_locking
            )
            update_beta2 = tf.assign(
                beta2_power, beta2_power * self._beta2_t, use_locking=self._use_locking
            )

        # Compute updated variables.
        updated_vars = []
        with tf.control_dependencies([update_beta1, update_beta2]):
            for var, var_update in zip(var_list, var_updates):
                assert var_updates is not None
                updated_vars.append(
                    (var - var_update) if var_update is not None else var
                )

    return updated_vars


def compute_sgd_updates(self, grads_and_vars):
    """Constructs and returns tensors of SGD-updated parameters."""
    grads_and_vars = tuple(grads_and_vars)
    var_list = [v for _, v in grads_and_vars]

    # Calculate SGD updates.
    with tf.name_scope(self._name):
        # Initialize.
        with tf.init_scope():
            self._create_slots(var_list)
        self._prepare()

        # Compute updates for each variable.
        var_updates = []
        for grad, var in grads_and_vars:
            if grad is None:
                var_updates.append(None)
                continue

            lr = tf.cast(self._learning_rate_tensor, var.dtype.base_dtype)
            var_updates.append(lr * grad)

        # Compute updated variables.
        updated_vars = []
        for var, var_update in zip(var_list, var_updates):
            assert var_updates is not None
            updated_vars.append((var - var_update) if var_update is not None else var)

    return updated_vars


def _create_non_slot_variable(self, initial_value, name, colocate_with):
    """Add an extra variable, not associated with a slot.
    Basically, remove the default manual colocation as it creates problems.
    """
    graph = colocate_with.graph

    key = (name, graph)
    v = self._non_slot_dict.get(key, None)
    if v is None:
        self._maybe_initialize_trackable()
        v = tf.Variable(initial_value, name=name, trainable=False)
        self._handle_deferred_dependencies(name=name, trackable=v)
        self._non_slot_dict[key] = v

    return v


# Monkey-patch optimizers.
tf.train.AdamOptimizer.compute_updates = compute_adam_updates
tf.train.GradientDescentOptimizer.compute_updates = compute_sgd_updates
tf.train.AdamOptimizer._create_non_slot_variable = _create_non_slot_variable
