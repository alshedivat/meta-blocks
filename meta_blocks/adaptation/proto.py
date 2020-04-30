"""Prototypical adaptation strategies."""

import logging

import tensorflow.compat.v1 as tf

from meta_blocks import common
from meta_blocks.adaptation import base
from meta_blocks.adaptation import proto_utils as utils

__all__ = ["Proto"]

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


class Proto(base.AdaptationStrategy):
    """Prototype-based model adaptation.

    Parameters
    ----------
    model : Model
        The model being adapted.

    optimizer : Optimizer
        The optimizer to use for meta-training.

    tasks : tuple of Tasks
        A tuple of tasks that provide access to data.

    mode : str, optional (default: common.ModeKeys.TRAIN)
        Defines the mode of the computation graph (TRAIN or EVAL).

    name : str, optional
        Name of the adaptation method.
    """

    def __init__(
        self,
        model,
        optimizer,
        tasks,
        mode=common.ModeKeys.TRAIN,
        name=None,
        **_unused_kwargs,
    ):
        super(Proto, self).__init__(
            model=model,
            optimizer=optimizer,
            tasks=tasks,
            mode=mode,
            name=(name or self.__class__.__name__),
        )

        # Inner loop.
        self.prototypes = None

    def _build_adapted_model(self, *, prototypes=None, task_id=None):
        """Returns a model with the specified prototypes."""
        if prototypes is None:
            prototypes = self.prototypes[task_id]
        self.model.prototypes = prototypes
        return self.model

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
        num_steps = num_classes
        # <int32> [num_steps, batch_size].
        batched_indices = tf.stack(tf.split(indices, num_steps, axis=0))

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

    def _build_adaptation(self):
        """Builds the adaptation loop."""
        self.prototypes = []
        for i, task in enumerate(self.tasks):
            # Build prototypes.
            inputs, labels = task.support_tensors
            self.prototypes.append(
                tf.cond(
                    pred=tf.not_equal(tf.size(labels), 0),
                    true_fn=lambda: self._build_prototypes(
                        inputs=inputs,
                        labels=labels,
                        num_classes=task.num_classes,
                        embedding_dim=self.model.embedding_dim,
                        back_prop=(self.mode == common.ModeKeys.TRAIN),
                    ),
                    # If no support data, use random prototypes.
                    false_fn=lambda: tf.random.normal(
                        shape=(task.num_classes, self.model.embedding_dim), stddev=1.0
                    ),
                )
            )
