"""Reptile-based adaptation strategies."""

import collections
import logging

import tensorflow.compat.v1 as tf

from meta_blocks import common
from meta_blocks.adaptation import maml

__all__ = ["Reptile"]

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


class Reptile(maml.Maml):
    """Reptile model adaptation.

    Algorithmically, adaptation is identical to first-order MAML.
    The difference is in how meta-update is performed.

    Parameters
    ----------
    model : Model
        The model being adapted.

    optimizer : Optimizer
        The optimizer to use for meta-training.

    tasks : tuple of Tasks
        A tuple of tasks that provide access to data.

    batch_size: int, optional (default: 16)
        Batch size used at adaptation time.

    inner_optimizer : Optimizer, optional (default: None)
        The optimizer to use for computing inner loop updates.

    mode : str, optional (default: common.ModeKeys.TRAIN)
        Defines the mode of the computation graph (TRAIN or EVAL).
        Note: this might be removed from the API down the line.

    name : str, optional (default: "Reptile")
        Name of the adaptation method.
    """

    def __init__(
        self,
        model,
        optimizer,
        tasks,
        batch_size=16,
        inner_optimizer=None,
        mode=common.ModeKeys.TRAIN,
        name="Reptile",
        **kwargs,
    ):
        # Instantiate Reptile.

        super(Reptile, self).__init__(
            model=model,
            optimizer=optimizer,
            tasks=tasks,
            batch_size=batch_size,
            inner_optimizer=inner_optimizer,
            first_order=True,
            mode=mode,
            name=name,
            **kwargs,
        )

    def _build_meta_train(self):
        """Internal fucntion for building meta-update op.
        """
        meta_grads = collections.defaultdict(list)
        with tf.name_scope("meta-learning"):
            # Build meta-loss.
            losses = self._build_meta_losses()
            meta_loss = tf.reduce_mean(losses)

            # Compute updates for the global parameters, if available.
            if self.model.global_parameters:
                for i, loss in enumerate(losses):
                    grads_and_vars = self.optimizer.compute_gradients(
                        loss, self.model.global_parameters
                    )
                    for g, v in grads_and_vars:
                        meta_grads[v].append(g)

            # Compute updates for the adaptable parameters.
            for i in range(len(self.tasks)):
                for name, value in self.model.adaptable_parameters.items():
                    value_upd = self._adapted_params[i][name]
                    meta_grads[value].append(value - value_upd)

            # Build meta-train op.
            meta_train_op = self.optimizer.apply_gradients(
                [(tf.reduce_mean(g, axis=0), v) for v, g in meta_grads.items()]
            )

            return meta_loss, meta_train_op

    def _build_adaptation(self):
        """Internal method for building adaption
        """
        # Placeholder for the number of adaptation steps.
        self._adapt_steps_ph = tf.placeholder(dtype=tf.int32)

        # Build adapted parameters.
        self._adapted_params = []
        for i, task in enumerate(self.tasks):
            inputs, labels = task.support_tensors
            if self.mode == common.ModeKeys.TRAIN:
                query_inputs, query_labels = task.query_tensors
                inputs = tf.concat([inputs, query_inputs], axis=0)
                labels = tf.concat([labels, query_labels], axis=0)
            self._adapted_params.append(
                tf.cond(
                    pred=tf.not_equal(tf.size(labels), 0),
                    true_fn=lambda: self._build_adapted_params(
                        inputs=inputs,
                        labels=labels,
                        init_params=self.model.adaptable_parameters,
                        num_steps=self._adapt_steps_ph,
                        back_prop=False,
                    ),
                    # If no support data, use initial parameters.
                    false_fn=lambda: self.model.adaptable_parameters,
                )
            )
