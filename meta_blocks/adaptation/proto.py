"""Proto(net) adaptation strategies."""

import logging

import tensorflow.compat.v1 as tf

from meta_blocks import common
from meta_blocks import models
from meta_blocks.adaptation import base
from meta_blocks.adaptation import utils

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
        Note: this might be removed from the API down the line.

    name : str, optional (default: "Proto")
        Name of the adaptation method.
    """

    def __init__(
        self,
        model,
        optimizer,
        tasks,
        mode=common.ModeKeys.TRAIN,
        name="Proto",
        **kwargs,
    ):
        # Instantiate Proto
        if not isinstance(model, models.base.ProtoModel):
            raise ValueError("Proto-based adaptation expects a ProtoModel.")
        super(Proto, self).__init__(
            model=model,
            optimizer=optimizer,
            tasks=tasks,
            mode=mode,
            name=name,
            **kwargs,
        )

        # Inner loop.
        self._prototypes = None

    def get_adapted_model(self, *, prototypes=None, task_id=None):
        """Returns a model with the specified prototypes."""
        if prototypes is None:
            prototypes = self._prototypes[task_id]
        self.model.prototypes = prototypes
        return self.model

    def _build_prototypes(
        self,
        inputs,
        labels,
        emb_size,
        num_classes,
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
            embeddings = self.model.build_embeddings(x)
            new_prototypes, new_class_counts = utils.build_prototypes(
                embeddings, y, num_classes
            )
            return [
                tf.add(step, 1),
                tf.add(prototypes, new_prototypes),
                tf.add(class_counts, new_class_counts),
            ]

        # Iterate through the data and compute average prototypes.
        proto_shape = (num_classes, emb_size)
        init_prototypes = tf.zeros(proto_shape)
        init_class_counts = tf.zeros(proto_shape[:1])
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

    def _build_meta_losses(self):
        """Builds meta-losses for each task."""
        meta_losses = []
        for i, task in enumerate(self.tasks):
            inputs, labels = task.query_tensors
            self.model.prototypes = self._prototypes[i]
            loss = self.model.build_loss(
                inputs, labels, num_classes=task.num_classes
            )
            meta_losses.append(loss)
        return meta_losses

    def _build_meta_eval(self):
        """Builds query predictions and labels."""
        preds_and_labels = []
        for i, task in enumerate(self.tasks):
            inputs, labels = task.query_tensors
            self.model.prototypes = self._prototypes[i]
            preds = self.model.build_predictions(inputs)
            preds_and_labels.append((preds, labels))
        return preds_and_labels

    def _build_adaptation(self):
        self._prototypes = []
        for i, task in enumerate(self.tasks):
            unlabeled_inputs = task.unlabeled_support_inputs
            emb_stddev = tf.math.reduce_std(
                tf.norm(self.model.build_embeddings(unlabeled_inputs), axis=-1)
            )
            # Build prototypes.
            inputs, labels = task.support_tensors
            self._prototypes.append(
                tf.cond(
                    pred=tf.not_equal(tf.size(labels), 0),
                    true_fn=lambda: self._build_prototypes(
                        inputs=inputs,
                        labels=labels,
                        emb_size=self.model.EMB_SIZE,
                        num_classes=task.num_classes,
                        back_prop=(self.mode == common.ModeKeys.TRAIN),
                    ),
                    # If no support data, use random prototypes.
                    false_fn=lambda: tf.random.normal(
                        shape=(task.num_classes, self.model.EMB_SIZE),
                        stddev=emb_stddev,
                    ),
                )
            )
