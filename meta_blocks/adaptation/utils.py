"""Adaptation utility functions."""

import logging

import tensorflow.compat.v1 as tf

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


def make_custom_getter(custom_variables):
    """Provides a custom getter for the given variables.

    The custom getter is such that whenever `get_variable` is called it
    will replace any trainable variables with the tensors in `variables`,
    in the same order. Non-trainable variables are obtained using the
    default getter for the current variable scope.

    Parameters
    ----------
    custom_variables : dict
        A dict of tensors replacing the trainable variables.

    Returns
    -------
    custom_getter : function
        A custom getter function that can be used at TF graph construction time.
    """

    def custom_getter(getter, name, **kwargs):
        if name in custom_variables:
            variable = custom_variables[name]
        else:
            variable = getter(name, **kwargs)
        return variable

    return custom_getter


def build_new_parameters(loss, params, optimizer, first_order=False):
    """Builds new parameters via an optimization step on the provided loss.

    Parameters
    ----------
    loss : <float32> [] tensor
        A scalar tensor that represents the loss.

    params : dict of variables or tensors
        A dictionary of initial parameters.

    optimizer : Optimizer
        An optimizer used for computing parameter updates.

    first_order : bool, optional (default: False)
        If True, gradients of the parameters computed by the optimizer are
        added to the graph as constants. This will zeros out the second order
        terms under subsequent differentiation.

    Returns
    -------
    new_params : dict of tensors
        A dictionary of update parameters.
    """
    param_names, param_values = zip(*params.items())
    grads_and_vars = optimizer.compute_gradients(loss, param_values)
    # Prevent backprop through the gradients, if necessary.
    if first_order:
        grads_and_vars = [(tf.stop_gradient(g), v) for g, v in grads_and_vars]
    new_params = dict(zip(param_names, optimizer.compute_updates(grads_and_vars)))
    return new_params


def build_prototypes(embeddings, labels, num_classes):
    """Builds new prototypes by aggregating embeddings.

    Parameters
    ----------
    embeddings : Tensor <float32> [num_inputs, emb_size]
        A collection of embeddings for each input point.

    labels : Tensor <int32> [num_inputs]
        Labels for each input point.

    num_classes : int
        The total number of classes.

    Returns
    -------
    prototypes : Tensor <float32> [num_classes, emb_size]
        A collection of prototypical embeddings for each class.
        Computed as a sum over embeddings of points corresponding to each class.

    class_counts : Tensor <float32> [num_classes].
        A vector representing the number of points of each class.
    """
    # <float32> [num_inputs, num_classes].
    labels_onehot = tf.one_hot(labels, num_classes)
    # <float32> [num_classes, emb_dim].
    prototypes = tf.einsum("ij,ik->kj", embeddings, labels_onehot)
    # <float32> [num_classes].
    class_counts = tf.reduce_sum(labels_onehot, axis=0)
    return prototypes, class_counts
