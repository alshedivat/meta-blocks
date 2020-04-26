"""MAML-based adaptation strategies."""

import logging

import tensorflow.compat.v1 as tf

from meta_blocks import common, optimizers
from meta_blocks.adaptation import base
from meta_blocks.adaptation import maml_utils as utils

__all__ = ["Maml"]

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


class Maml(base.AdaptationStrategy):
    """MAML-based model adaptation.

    Parameters
    ----------
    model : Model
        The model being adapted.

    optimizer : Optimizer
        The optimizer to use for meta-training.

    tasks : tuple of Tasks
        A tuple of tasks that provide access to data.

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

    name : str, optional (default: "Maml")
        Name of the adaptation method.
    """

    def __init__(
        self,
        model,
        optimizer,
        tasks,
        batch_size=None,
        num_inner_steps=1,
        inner_optimizer=None,
        first_order=False,
        mode=common.ModeKeys.TRAIN,
        name=None,
        **_unused_kwargs,
    ):
        super(Maml, self).__init__(
            model=model,
            optimizer=optimizer,
            tasks=tasks,
            mode=mode,
            name=(name or self.__class__.__name__),
        )
        self.batch_size = batch_size
        self.num_inner_steps = num_inner_steps
        self.first_order = first_order

        # Inner loop.
        self.inner_optimizer = optimizers.get(**(inner_optimizer or {}))
        self.adapted_parameters = None

    def _build_adapted_model(self, *, parameters=None, task_id=None):
        """Builds a model with the specified adapted parameters."""
        if parameters is None:
            parameters = self.adapted_parameters[task_id]
        with utils.custom_make_variable(parameters, outer_scope=self.model.name):
            adapted_model = self.model.build()
        return adapted_model

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
            # Build the loss with custom parameters.
            adapted_model = self._build_adapted_model(parameters=parameters)
            loss = adapted_model.build_loss(x, y)
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

    def _build_adaptation(self):
        """Builds the adaptation loop."""
        # Initial parameters.
        self.initial_parameters = {}
        for param in self.model.initial_parameters:
            canonical_name = utils.canonical_variable_name(
                param.name, outer_scope=self.model.name
            )
            self.initial_parameters[canonical_name] = param
        # Build new adapted params.
        self.adapted_parameters = []
        for i, task in enumerate(self.tasks):
            inputs, labels = task.support_tensors
            with tf.name_scope(f"task{i}"):
                self.adapted_parameters.append(
                    tf.cond(
                        pred=tf.not_equal(tf.size(labels), 0),
                        true_fn=lambda: self._build_adapted_parameters(
                            inputs=inputs,
                            labels=labels,
                            initial_parameters=self.initial_parameters,
                            num_steps=self.num_inner_steps,
                            back_prop=(self.mode == common.ModeKeys.TRAIN),
                        ),
                        # If support data is empty, use initial parameters.
                        false_fn=lambda: self.initial_parameters,
                    )
                )
