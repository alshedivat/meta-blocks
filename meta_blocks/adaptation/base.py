"""Base classes and functionality for model adaptation."""

import abc
import logging

import tensorflow.compat.v1 as tf

__all__ = ["AdaptationStrategy"]

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


class AdaptationStrategy(abc.ABC):
    """Base base class for meta-trainable adaptation strategies.

    Parameters
    ----------
    model : Model
        The model being adapted.

    optimizer : Optimizer
        The optimizer used for computing meta-updates.

    tasks : tuple of Tasks
        A tuple of tasks that provide access to data.

    mode : str, optional (default: common.ModeKeys.TRAIN)
        Defines the mode of the computation graph (TRAIN or EVAL).

    name : str, optional
        Name of the adaptation method.
    """

    def __init__(self, model, optimizer, tasks, mode=None, name=None):
        self.model = model
        self.optimizer = optimizer
        self.tasks = tasks

        # Internals.
        self.loss = None
        self.train_op = None
        self.preds_and_labels = None

        # Misc.
        self.mode = mode
        self.name = name or self.__class__.__name__
        self.built = False

    @abc.abstractmethod
    def _build_adapted_model(self, **kwargs):
        """Builds adapted model graph. Must be implemented in a subclass."""
        raise NotImplementedError("Abstract Method")

    @abc.abstractmethod
    def _build_adaptation(self):
        """Builds the adaptation graph. Must be implemented in a subclass."""
        raise NotImplementedError("Abstract Method")

    def _build_meta_losses(self):
        """Builds meta-losses for each task."""
        meta_losses = []
        for i, task in enumerate(self.tasks):
            inputs, labels = task.query_tensors
            with tf.name_scope(f"task{i}"):
                adapted_model = self._build_adapted_model(task_id=i)
                loss = adapted_model.build_loss(inputs, labels)
                meta_losses.append(loss)
        return meta_losses

    def _build_meta_learn(self):
        """Builds meta-learning ops."""
        meta_losses, preds_and_labels = [], []
        for i, task in enumerate(self.tasks):
            query_inputs, query_labels = task.query_tensors
            with tf.name_scope(f"task{i}"):
                adapted_model = self._build_adapted_model(task_id=i)
                loss = adapted_model.build_loss(query_inputs, query_labels)
                query_preds = adapted_model.build_predictions(query_inputs)
                preds_and_labels.append((query_preds, query_labels))
                meta_losses.append(loss)
        # Build meta-loss and meta-train op.
        meta_loss = tf.reduce_mean(meta_losses)
        meta_train_op = self.optimizer.minimize(
            meta_loss, var_list=self.model.initial_parameters
        )
        return meta_loss, meta_train_op, preds_and_labels

    def build(self):
        """Builds the graph for adaptation strategy."""
        logger.debug(f"Building {self.name}...")
        with tf.name_scope(self.name):
            # Build adaptation.
            logger.debug("Building adaptation graph...")
            with tf.name_scope("adaptation"):
                self._build_adaptation()
            # Build meta-losses, meta-update ops, meta-eval.
            logger.debug("Building meta-learning ops...")
            with tf.name_scope("meta-learn"):
                (
                    self.loss,
                    self.train_op,
                    self.preds_and_labels,
                ) = self._build_meta_learn()
            logger.debug("Done.")
        return self
