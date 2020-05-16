"""Reptile-based adaptation strategies."""

import collections
import logging
from typing import Callable, List, Optional

import tensorflow.compat.v1 as tf

from meta_blocks import common
from meta_blocks.adaptation import maml
from meta_blocks.adaptation import maml_utils as utils
from meta_blocks.tasks.base import Task, TaskDistribution

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
    model_builder : Callable
        A builder function for the model.

    optimizer : Optimizer
        The optimizer to use for meta-training.

    task_dists : list of TaskDistributions
        A list of task distributions with which meta-learner interacts.

    batch_size: int, optional
        Batch size used at adaptation time.

    num_inner_steps : int, optional (default: 10)
        Number of inner adaptation steps.

    inner_optimizer : Optimizer, optional (default: None)
        The optimizer to use for computing inner loop updates.

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
        num_inner_steps: int = 10,
        inner_optimizer: Optional[dict] = None,
        mode: str = common.ModeKeys.TRAIN,
        name: Optional[str] = None,
        **_unused_kwargs,
    ):
        super(Reptile, self).__init__(
            model_builder=model_builder,
            optimizer=optimizer,
            task_dists=task_dists,
            batch_size=batch_size,
            num_inner_steps=num_inner_steps,
            inner_optimizer=inner_optimizer,
            first_order=True,
            mode=mode,
            name=(name or self.__class__.__name__),
        )

    # --- Methods. ---

    def _build_meta_train_ops(self):
        """Builds meta-update op."""
        # Reptile does not have a proper meta-loss.
        self.meta_losses = [tf.constant(-1.0) for _ in self.task_dists]
        # Compute Reptile's meta-gradients.
        # Note: meta-gradients are computed as the difference between the
        #       initial and adapted model parameters.
        meta_grads = collections.defaultdict(list)
        for td, td_adapted_models in zip(self.task_dists, self.adapted_models):
            with tf.name_scope(td.name):
                for task, adapted_model in zip(td.task_batch, td_adapted_models):
                    with tf.name_scope(task.name):
                        initial_params = self.model.trainable_parameters
                        adapted_params = adapted_model.trainable_parameters
                        for p, p_upd in zip(initial_params, adapted_params):
                            meta_grads[p].append(p - p_upd)
        # Build meta-train op.
        self.meta_train_op = self.optimizer.apply_gradients(
            [(tf.reduce_mean(g, axis=0), v) for v, g in meta_grads.items()]
        )

    def build_adapted_model(self, task: Task):
        """Builds a model with gradient-based adapted parameters."""
        # Initial parameters.
        initial_parameters = {}
        for param in self.model.trainable_parameters:
            canonical_name = utils.canonical_variable_name(
                param.name, outer_scope=self.model.name
            )
            initial_parameters[canonical_name] = param
        # Build adapted parameters.
        inputs, labels = task.support_tensors
        # Reptile does not distinguish between support and query sets.
        if self.mode == common.ModeKeys.TRAIN:
            query_inputs, query_labels = task.query_tensors
            inputs = tf.concat([inputs, query_inputs], axis=0)
            labels = tf.concat([labels, query_labels], axis=0)
        adapted_parameters = tf.cond(
            pred=tf.not_equal(tf.size(labels), 0),
            true_fn=lambda: self._build_adapted_parameters(
                inputs=inputs,
                labels=labels,
                initial_parameters=initial_parameters,
                num_steps=self.num_inner_steps,
                back_prop=False,
            ),
            # If support data is empty, use initial parameters.
            false_fn=lambda: initial_parameters,
        )
        # Build adapted model.
        with utils.custom_make_variable(adapted_parameters, self.model.name):
            adapted_model = self.model_builder()
        return adapted_model
