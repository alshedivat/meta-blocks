"""Adaptation utility functions."""

import contextlib
import logging
from typing import Dict

import tensorflow.compat.v1 as tf
from tensorflow.python import ops
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine.base_layer_utils import (
    make_variable as original_make_variable,
)

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


def canonical_variable_name(variable_name: str, outer_scope: str):
    """Returns the canonical variable name: `outer_scope/.../name`."""
    name_parts = variable_name.split(":")[0].split("/")
    for i, part in enumerate(name_parts):
        if part == outer_scope:
            return "/".join(name_parts[i:])
    raise ValueError(f"Variable <{variable_name}> does not belong to <{outer_scope}>.")


@contextlib.contextmanager
def custom_make_variable(
    canonical_custom_variables: Dict[str, tf.Tensor], outer_scope: str
):
    """A context manager that overrides `make_variable` with a custom function.

    When building layers, Keras uses `make_variable` function to create weights
    (kernels and biases for each layer). To build computation graphs that
    consist of layers with adapted weights without implementing a whole new
    custom library of layers, this function directly patches Keras internals.
    Currently, this is the only solution compatible with Keras and TF2.
    See: https://github.com/tensorflow/tensorflow/issues/33125.

    This function wraps `make_variable` with a closure that infers the canonical
    name of the variable being created (of the form `ModelName/.../var_name`)
    and looks it up in the `custom_variables` dict that maps canonical names
    to tensors. The function adheres the following logic:
    - If there is a match, it does a few checks (shape, dtype, etc.) and returns
      the found tensor instead of creating a new variable.
    - If there is a match but checks fail, it throws an exception.
    - If there are no matching `custom_variables`, it calls the original
      `make_variable` utility function and returns a newly created variable.

    Parameters
    ----------
    canonical_custom_variables : dict
        A dict of canonical variable names to tensors that must replace them.

    outer_scope : str
        Name of the scope under which all variables are built.
        This is usually the name of the model.
    """

    def _custom_make_variable(name, **kwargs):
        # Create a variable using the original variable getter.
        variable_name = f"{ops.get_name_scope()}/{name}"
        canonical_name = canonical_variable_name(variable_name, outer_scope)
        if canonical_name in canonical_custom_variables:
            custom_variable = canonical_custom_variables[canonical_name]
            # Check that custom_variable is a valid replacement.
            if (
                kwargs["shape"] != custom_variable.shape
                or kwargs["dtype"] != custom_variable.dtype
            ):
                # TODO: raise a more specific exception.
                raise Exception(
                    f"{custom_variable} cannot replace for <{variable_name}>."
                )
            return custom_variable
        else:
            variable = original_make_variable(name=name, **kwargs)
        return variable

    try:
        # Monkey-patch make_variable.
        base_layer_utils.make_variable = _custom_make_variable
        yield
    finally:
        # Monkey-unpatch make_variable.
        base_layer_utils.make_variable = original_make_variable


def build_new_parameters(loss, parameters, optimizer, first_order=False):
    """Builds new parameters via an optimization step on the provided loss.

    Parameters
    ----------
    loss : <float32> [] tensor
        A scalar tensor that represents the loss.

    parameters : dict of variables or tensors
        A dictionary of initial parameters.

    optimizer : Optimizer
        An optimizer used for computing parameter updates.

    first_order : bool, optional (default: False)
        If True, gradients of the parameters computed by the optimizer are
        added to the graph as constants. This will zeros out the second order
        terms under subsequent differentiation.

    Returns
    -------
    new_parameters : dict of tensors
        A dictionary of update parameters.
    """
    param_names, param_values = zip(*parameters.items())
    grads_and_vars = optimizer.compute_gradients(loss, param_values)
    # Prevent backprop through the gradients, if necessary.
    if first_order:
        grads_and_vars = [(tf.stop_gradient(g), v) for g, v in grads_and_vars]
    new_parameters = dict(zip(param_names, optimizer.compute_updates(grads_and_vars)))
    return new_parameters


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
