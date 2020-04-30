"""Reptile-based adaptation strategies."""

import collections
import logging

import tensorflow.compat.v1 as tf

from meta_blocks import common
from meta_blocks.adaptation import maml
from meta_blocks.adaptation import maml_utils as utils

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

    batch_size: int, optional
        Batch size used at adaptation time.

    num_inner_steps : int, optional (default: 10)
        Number of inner adaptation steps.

    inner_optimizer : Optimizer, optional (default: None)
        The optimizer to use for computing inner loop updates.

    mode : str, optional (default: common.ModeKeys.TRAIN)
        Defines the mode of the computation graph (TRAIN or EVAL).

    name : str, optional (default: "Reptile")
        Name of the adaptation method.
    """

    def __init__(
        self,
        model,
        optimizer,
        tasks,
        batch_size=None,
        num_inner_steps=10,
        inner_optimizer=None,
        mode=common.ModeKeys.TRAIN,
        name=None,
        **_unused_kwargs,
    ):
        super(Reptile, self).__init__(
            model=model,
            optimizer=optimizer,
            tasks=tasks,
            batch_size=batch_size,
            num_inner_steps=num_inner_steps,
            inner_optimizer=inner_optimizer,
            first_order=True,
            mode=mode,
            name=(name or self.__class__.__name__),
        )

    def _build_meta_learn(self):
        """Builds meta-update op."""
        # Reptile does not have a proper meta-loss.
        meta_loss = tf.constant(-1.0)
        # Compute meta-gradients and predictions.
        preds_and_labels = []
        meta_grads = collections.defaultdict(list)
        for i, task in enumerate(self.tasks):
            query_inputs, query_labels = task.query_tensors
            with tf.name_scope(f"task{i}"):
                for name, value in self.initial_parameters.items():
                    value_upd = self.adapted_parameters[i][name]
                    meta_grads[value].append(value - value_upd)
                adapted_model = self._build_adapted_model(task_id=i)
                query_preds = adapted_model.build_predictions(query_inputs)
                preds_and_labels.append((query_preds, query_labels))
        # Build meta-train op.
        meta_train_op = self.optimizer.apply_gradients(
            [(tf.reduce_mean(g, axis=0), v) for v, g in meta_grads.items()]
        )
        return meta_loss, meta_train_op, preds_and_labels

    def _build_adaptation(self):
        """Builds the adaptation loop."""
        # Initial parameters.
        self.initial_parameters = {}
        for param in self.model.initial_parameters:
            canonical_name = utils.canonical_variable_name(
                param.name, outer_scope=self.model.name
            )
            self.initial_parameters[canonical_name] = param
        # Build adapted parameters.
        self.adapted_parameters = []
        for i, task in enumerate(self.tasks):
            inputs, labels = task.support_tensors
            with tf.name_scope(f"task{i}"):
                # Reptile does not distinguish between support and query sets.
                if self.mode == common.ModeKeys.TRAIN:
                    query_inputs, query_labels = task.query_tensors
                    inputs = tf.concat([inputs, query_inputs], axis=0)
                    labels = tf.concat([labels, query_labels], axis=0)
                self.adapted_parameters.append(
                    tf.cond(
                        pred=tf.not_equal(tf.size(labels), 0),
                        true_fn=lambda: self._build_adapted_parameters(
                            inputs=inputs,
                            labels=labels,
                            initial_parameters=self.initial_parameters,
                            num_steps=self.num_inner_steps,
                            back_prop=False,
                        ),
                        # If support data is empty, use initial parameters.
                        false_fn=lambda: self.initial_parameters,
                    )
                )
