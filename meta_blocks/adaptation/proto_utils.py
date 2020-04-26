"""Adaptation utility functions."""

import logging

import tensorflow.compat.v1 as tf

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


def build_prototypes(embeddings: tf.Tensor, labels: tf.Tensor, num_classes: int):
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
