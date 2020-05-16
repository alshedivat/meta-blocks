"""MAML-based adaptation strategies."""

import logging
from typing import Callable, List, Optional

import tensorflow.compat.v1 as tf

from meta_blocks import common, optimizers
from meta_blocks.adaptation import base
from meta_blocks.adaptation import maml_utils as utils
from meta_blocks.tasks.base import Task, TaskDistribution

__all__ = ["Maml"]

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


class Maml(base.MetaLearner):
    """MAML-based model adaptation.

    Parameters
    ----------
    model_builder : Callable
        A builder function for the model.

    optimizer : Optimizer
        The optimizer to use for meta-training.

    task_dists : list of TaskDistributions
        A list of task distributions with which meta-learner interacts.

    batch_size : int, optional
        Batch size used at adaptation time.

    num_inner_steps : int, optional (default: 1)
        Number of inner adaptation steps.

    inner_optimizer : Optimizer, optional (default: None)
        The optimizer to use for computing inner loop updates.

    first_order : bool, optional (default: False)
        If True, drops second order terms in the meta-updates.

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
        num_inner_steps: int = 1,
        inner_optimizer: Optional[dict] = None,
        first_order: bool = False,
        mode: str = common.ModeKeys.TRAIN,
        name: Optional[str] = None,
        **_unused_kwargs,
    ):
        self.batch_size = batch_size
        self.num_inner_steps = num_inner_steps
        self.first_order = first_order

        # Inner loop.
        self.inner_optimizer = optimizers.get(**(inner_optimizer or {}))
        self.inner_adapted_models = []

        super(Maml, self).__init__(
            model_builder=model_builder,
            optimizer=optimizer,
            task_dists=task_dists,
            mode=mode,
            name=(name or self.__class__.__name__),
        )

    # --- Properties. ---

    @property
    def non_trainable_parameters(self):
        """Collects and returns all non-trainable variables."""
        non_trainable = super(Maml, self).non_trainable_parameters
        # Collect non-trainable variables from the inner loop models.
        for adapted_model in self.inner_adapted_models:
            non_trainable += adapted_model.non_trainable_parameters
        return non_trainable

    # --- Methods. ---

    def _build_adapted_parameters(
        self,
        inputs,
        labels,
        initial_parameters,
        num_steps,
        back_prop=False,
        parallel_iterations=1,
        shuffle=True,
    ):
        """Builds adapted model parameters dynamically using tf.while_loop.

        Parameters
        ----------
        inputs : Tensor <float32> [None, ...]
            Inputs of the samples used for building adapted parameters.

        labels : Tensor <float32> [None, num_classes]
            Labels of the samples used for building adapted parameters.

        initial_parameters : dict of Tensors
            A dictionary with initial parameters of the model.

        num_steps : int or Tensor <int32> []
            Number of gradient steps used for adaptation.

        back_prop : bool, optional (default: False)
            Indicates whether backprop is allowed through the adapted parameters.

        parallel_iterations : int, optional (default=1)
            Parallel iterations parameter for the tf.while_loop.

        shuffle : bool, optional (default=True)
            Whether to shuffle the samples before batching.

        Returns
        -------
        adapted_parameters : dict of Tensors
            A dictionary with adapted parameters of the model.
        """
        # If batch size not specified, use all inputs.
        batch_size = self.batch_size or tf.shape(inputs)[0]
        # Build batched indices.
        # <int32> [batch_size * num_steps].
        indices = tf.math.mod(
            tf.range(batch_size * num_steps, dtype=tf.int32), tf.shape(inputs)[0]
        )
        if shuffle:
            indices = tf.random.shuffle(indices)
        # <int32> [num_steps, batch_size].
        batched_indices = tf.reshape(indices, shape=(num_steps, batch_size))

        def cond_fn(step, _unused_params):
            return tf.less(step, num_steps)

        def body_fn(step, parameters):
            x = tf.gather(inputs, batched_indices[step], axis=0)
            y = tf.gather(labels, batched_indices[step], axis=0)
            # Build a model with new parameters.
            with utils.custom_make_variable(parameters, self.model.name):
                self.inner_adapted_models.append(self.model_builder())
            loss = self.inner_adapted_models[-1].loss(x, y)
            # Build new parameters.
            new_parameters = utils.build_new_parameters(
                loss,
                parameters,
                optimizer=self.inner_optimizer,
                first_order=self.first_order,
            )
            return [tf.add(step, 1), new_parameters]

        _, adapted_parameters = tf.while_loop(
            cond=cond_fn,
            body=body_fn,
            loop_vars=[tf.constant(0), initial_parameters],
            parallel_iterations=parallel_iterations,
            back_prop=back_prop,
            name="adapt",
        )

        return adapted_parameters

    def build_adapted_model(self, task: Task):
        """Builds a model with gradient-based adapted parameters."""
        # Initial parameters.
        initial_parameters = {}
        for parameter in self.model.trainable_parameters:
            canonical_name = utils.canonical_variable_name(
                parameter.name, outer_scope=self.model.name
            )
            initial_parameters[canonical_name] = parameter
        # Build adapted parameters.
        inputs, labels = task.support_tensors
        adapted_parameters = tf.cond(
            pred=tf.not_equal(tf.size(labels), 0),
            true_fn=lambda: self._build_adapted_parameters(
                inputs=inputs,
                labels=labels,
                initial_parameters=initial_parameters,
                num_steps=self.num_inner_steps,
                back_prop=(self.mode == common.ModeKeys.TRAIN),
            ),
            # If support data is empty, use initial parameters.
            false_fn=lambda: initial_parameters,
        )
        # Build adapted model.
        with utils.custom_make_variable(adapted_parameters, self.model.name):
            adapted_model = self.model_builder()
        return adapted_model
