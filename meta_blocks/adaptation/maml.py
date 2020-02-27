"""MAML-based adaptation strategies."""

import logging

import tensorflow.compat.v1 as tf

from .. import common
from .. import optimizers
from . import base
from . import utils

__all__ = ["Maml"]

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


class Maml(base.AdaptationStrategy):
    """MAML-based model adaptation."""

    def __init__(
        self,
        model,
        optimizer,
        tasks,
        batch_size=16,
        inner_optimizer=None,
        first_order=False,
        mode=common.ModeKeys.TRAIN,
        name="Maml",
        **kwargs,
    ):
        """Instantiate Maml.

        Args:
            model : Model
                The model being adapted.
            optimizer : Optimizer
                The optimizer to use for meta-training.
            tasks : tuple of Tasks
                A tuple of tasks that provide access to data.
            batch_size : int (default: 16)
                Batch size used at adaptation time.
            inner_optimizer : Optimizer (default: None)
                The optimizer to use for inner loop updates
            first_order : bool (default: False)
            mode : str (default: common.ModeKeys.TRAIN)
            name : str (default: "Maml")
            **kwargs
        """
        super(Maml, self).__init__(
            model=model,
            optimizer=optimizer,
            tasks=tasks,
            mode=mode,
            name=name,
            **kwargs,
        )
        self._batch_size = batch_size
        self._first_order = first_order

        # Inner loop.
        self._inner_optimizer = optimizers.get(**inner_optimizer)
        self._adapt_steps_ph = None
        self._adapted_params = None

    def get_adapted_model(self, *, params=None, task_id=None):
        """Returns a model with the specified adapted parameters."""
        if params is None:
            params = self._adapted_params[task_id]
        self.model.custom_getter = utils.make_custom_getter(params)
        return self.model

    def get_feed_list(self, num_inner_steps=1, **kwargs):
        """Constructs a feed list from the arguments."""
        return [(self._adapt_steps_ph, num_inner_steps)]

    def _build_adapted_params(
        self,
        inputs,
        labels,
        init_params,
        num_steps,
        back_prop=False,
        parallel_iterations=1,
        shuffle=True
    ):
        """Builds adapted model parameters dynamically using tf.while_loop.

        Args:
            inputs: tf.Tensor
            labels: tf.Tensor
            init_params: dict of tf.Tensors
            num_steps: int or tf.Tensor <int32> []
            back_prop: bool (default: False)
            parallel_iterations: int (default: 1)
            shuffle: bool (default: True)

        Returns:
            adapted_params: dict of tf.Tensors
        """
        # Build batched indices.
        # <int32> [batch_size * num_steps].
        indices = tf.math.mod(
            tf.range(self._batch_size * num_steps, dtype=tf.int32),
            tf.shape(inputs)[0]
        )
        if shuffle:
            indices = tf.random.shuffle(indices)
        # <int32> [num_steps, batch_size].
        batched_indices = tf.reshape(
            indices, shape=(num_steps, self._batch_size)
        )

        def cond_fn(step, _unused_params):
            return tf.less(step, num_steps)

        def body_fn(step, params):
            x = tf.gather(inputs, batched_indices[step], axis=0)
            y = tf.gather(labels, batched_indices[step], axis=0)
            # Build the loss with custom parameters.
            adapted_model = self.get_adapted_model(params=params)
            loss = adapted_model.build_loss(x, y)
            # Build new parameters.
            new_params = utils.build_new_parameters(
                loss,
                params,
                optimizer=self._inner_optimizer,
                first_order=self._first_order,
            )
            return [tf.add(step, 1), new_params]

        _, adapted_params = tf.while_loop(
            cond=cond_fn,
            body=body_fn,
            loop_vars=[tf.constant(0), init_params],
            parallel_iterations=parallel_iterations,
            back_prop=back_prop,
            name="adapt",
        )

        return adapted_params

    def _build_meta_losses(self):
        """Builds meta-losses for each task."""
        meta_losses = []
        for i, task in enumerate(self.tasks):
            inputs, labels = task.query_tensors
            adapted_model = self.get_adapted_model(task_id=i)
            loss = adapted_model.build_loss(inputs, labels)
            meta_losses.append(loss)
        return meta_losses

    def _build_meta_eval(self):
        """Builds query predictions and labels."""
        preds_and_labels = []
        for i, task in enumerate(self.tasks):
            inputs, labels = task.query_tensors
            adapted_model = self.get_adapted_model(task_id=i)
            preds = adapted_model.build_predictions(inputs)
            preds_and_labels.append((preds, labels))
        return preds_and_labels

    def _build_adaptation(self):
        """Builds the adaptation loop."""
        # Placeholder for the number of adaptation steps.
        self._adapt_steps_ph = tf.placeholder(
            dtype=tf.int32, shape=[], name="num_inner_steps"
        )

        # Build new adapted params.
        self._adapted_params = []
        for i, task in enumerate(self.tasks):
            inputs, labels = task.support_tensors
            self._adapted_params.append(
                tf.cond(
                    pred=tf.not_equal(tf.size(labels), 0),
                    true_fn=lambda: self._build_adapted_params(
                        inputs=inputs,
                        labels=labels,
                        init_params=self.model.adaptable_parameters,
                        num_steps=self._adapt_steps_ph,
                        back_prop=(self.mode == common.ModeKeys.TRAIN),
                    ),
                    # If no support data, use initial parameters.
                    false_fn=lambda: self.model.adaptable_parameters,
                )
            )
