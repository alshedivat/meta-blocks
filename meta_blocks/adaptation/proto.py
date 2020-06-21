"""Prototypical adaptation strategies."""

import copy
import logging
from typing import Callable, List, Optional

import tensorflow.compat.v1 as tf

from meta_blocks import common
from meta_blocks.adaptation import base
from meta_blocks.adaptation import proto_utils as utils
from meta_blocks.tasks.base import Task, TaskDistribution

__all__ = ["Proto"]

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


class Proto(base.MetaLearner):
    """Prototype-based model adaptation.

    Parameters
    ----------
    model_builder : Callable
        A builder function for the model.

    optimizer : Optimizer
        The optimizer to use for meta-training.

    task_dists : list of TaskDistributions
        A list of task distributions with which meta-learner interacts.

    mode : str, optional (default: common.ModeKeys.TRAIN)
        Defines the mode of the computation graph (TRAIN or EVAL).

    name : str, optional
        Name of the adaptation method.
    """

    def __init__(
        self,
        model_builder: Callable,
        optimizer: tf.train.Optimizer,
        task_dists: List[TaskDistribution],
        batch_size: Optional[int] = None,
        mode: str = common.ModeKeys.TRAIN,
        name: Optional[str] = None,
        **_unused_kwargs,
    ):
        self.batch_size = batch_size

        super(Proto, self).__init__(
            model_builder=model_builder,
            optimizer=optimizer,
            task_dists=task_dists,
            mode=mode,
            name=(name or self.__class__.__name__),
        )

    # --- Methods. ---

    def _build_prototypes(
        self,
        inputs,
        labels,
        num_classes,
        embedding_dim,
        back_prop=True,
        parallel_iterations=16,
        eps=1e-7,
    ):
        """Builds adapted model parameters dynamically using tf.while_loop."""
        indices = tf.range(tf.shape(inputs)[0])
        # Note: We assume that the number of given data points is proportional
        #       to the number of classes in the data, so we can split it into
        #       the corresponding number of batches.
        # TODO: make it work with imbalanced classes.
        num_steps = num_classes
        # Batched indies: <int32> [num_steps, batch_size].
        # batched_indices = tf.stack(tf.split(indices, num_steps, axis=0))
        if self.batch_size is not None:
            batch_size = self.batch_size
            num_steps = tf.cast(tf.shape(indices)[0] / batch_size, tf.int32)
        else:
            num_steps = num_classes
            batch_size = tf.cast(tf.shape(indices)[0] / num_steps, tf.int32)
        batched_indices = tf.reshape(indices, shape=(num_steps, batch_size))

        def cond_fn(step, _unused_prototypes, _unused_class_counts):
            return tf.less(step, num_steps)

        def body_fn(step, prototypes, class_counts):
            x = tf.gather(inputs, batched_indices[step], axis=0)
            y = tf.gather(labels, batched_indices[step], axis=0)
            embeddings = self.model.network(x)
            new_prototypes, new_class_counts = utils.build_prototypes(
                embeddings, y, num_classes
            )
            return [
                tf.add(step, 1),
                tf.add(prototypes, new_prototypes),
                tf.add(class_counts, new_class_counts),
            ]

        # Iterate through the data and compute average prototypes.
        prototypes_shape = (num_classes, embedding_dim)
        init_prototypes = tf.zeros(prototypes_shape)
        init_class_counts = tf.zeros(prototypes_shape[:1])
        _, cum_prototypes, cum_class_counts = tf.while_loop(
            cond=cond_fn,
            body=body_fn,
            loop_vars=[tf.constant(0), init_prototypes, init_class_counts],
            parallel_iterations=parallel_iterations,
            back_prop=back_prop,
            name="adapt",
        )

        # Compute average prototypes for each class.
        # <float32> [num_classes, 1].
        cum_class_counts_eps = tf.expand_dims(cum_class_counts + eps, -1)
        # <float32> [num_classes, emb_dim].
        return tf.math.divide(cum_prototypes, cum_class_counts_eps)

    def build_adapted_model(self, task: Task):
        """Builds a model with task-specific prototypes."""
        # Build prototypes.
        inputs, labels = task.support_tensors
        prototypes = tf.cond(
            pred=tf.not_equal(tf.size(labels), 0),
            true_fn=lambda: self._build_prototypes(
                inputs=inputs,
                labels=labels,
                num_classes=task.num_ways,
                embedding_dim=self.model.embedding_dim,
                back_prop=(self.mode == common.ModeKeys.TRAIN),
            ),
            # If no support data, use random prototypes.
            false_fn=lambda: tf.random.normal(
                shape=(task.num_ways, self.model.embedding_dim), stddev=1.0
            ),
        )
        # Build adapted model.
        assert self.model.prototypes is None
        adapted_model = copy.copy(self.model)
        adapted_model.prototypes = prototypes
        return adapted_model
