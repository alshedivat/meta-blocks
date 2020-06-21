"""Base classes and functionality for model adaptation."""

import abc
import logging
from typing import Callable, List, Optional

import tensorflow.compat.v1 as tf

from meta_blocks.tasks.base import Task, TaskDistribution

__all__ = ["MetaLearner"]

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


class MetaLearner(abc.ABC):
    """The base class for learning-to-learn algorithms.

    Subclasses must implement the `build_adapted_model` method that returns
    an adapted :class:`ClassificationModel` instance. Subclasses can also
    override _build_meta_train_ops and _build_meta_eval_ops if necessary.
    (For instance, see how Reptile overrides _build_meta_train_ops.)

    Parameters
    ----------
    model_builder : Callable
        A builder function for the model.

    optimizer : Optimizer
        The optimizer used for computing meta-updates.

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
        mode: Optional[str] = None,
        name: Optional[str] = None,
    ):
        self.model_builder = model_builder
        self.optimizer = optimizer
        self.task_dists = task_dists
        self.mode = mode
        self.name = name or self.__class__.__name__

        # Internals.
        self.meta_losses = None
        self.meta_train_op = None
        self.adapted_models = {}

        # Build the graph.
        logger.debug(f"Building {self.name}...")
        with tf.name_scope(self.name):
            # Build initial model.
            logger.debug("Building initial model...")
            with tf.name_scope("initial"):
                self.model = self.model_builder()
            # Build adapted model for each task.
            logger.debug("Building adapted models...")
            with tf.name_scope("adapted"):
                for td in self.task_dists:
                    td_adapted_models = {}
                    with tf.name_scope(td.name):
                        for i, task in enumerate(td.task_batch):
                            with tf.name_scope(task.name):
                                adapted_model = self.build_adapted_model(task)
                            td_adapted_models[task.name] = adapted_model
                    self.adapted_models[td.name] = td_adapted_models
            # Build losses, predictions, and meta-training ops.
            logger.debug("Building meta-learning ops...")
            with tf.name_scope("meta-learn"):
                self._build_meta_eval_ops()
                self._build_meta_train_ops()
            logger.debug("Done.")

    # --- Properties. ---

    @property
    def parameters(self):
        """Collects and returns all variables."""
        return self.trainable_parameters + self.non_trainable_parameters

    @property
    def trainable_parameters(self):
        """Collects and returns all trainable variables."""
        return self.model.trainable_parameters

    @property
    def non_trainable_parameters(self):
        """Collects and returns all non-trainable variables."""
        non_trainable = self.model.non_trainable_parameters
        for td in self.task_dists:
            for task in td.task_batch:
                adapted_model = self.get_adapted_model(td.name, task.name)
                non_trainable += adapted_model.non_trainable_parameters
        return non_trainable

    # --- Methods. ---

    def get_adapted_model(self, task_dist_name: str, task_name: str):
        """Returns the adapted model for the specified task."""
        return self.adapted_models[task_dist_name][task_name]

    def _build_meta_eval_ops(self):
        """Builds pre-/post-adaptation support/query losses and predictions.
        All losses and preds_and_labels are built for each task.
        """
        self.losses, self.preds_and_labels = [], []
        # Iterate over task distributions.
        for td in self.task_dists:
            td_losses, td_preds_and_labels = [], []
            with tf.name_scope(td.name):
                # Iterate over tasks in the batch within the distribution.
                for task in td.task_batch:
                    adapted = self.get_adapted_model(td.name, task.name)
                    inputs_s, labels_s = task.support_tensors
                    inputs_q, labels_q = task.query_tensors
                    # Build losses and predictions.
                    with tf.name_scope(task.name):
                        task_losses = {
                            "pre": {
                                "supp": self.model.loss(inputs_s, labels_s),
                                "query": self.model.loss(inputs_q, labels_q),
                            },
                            "post": {
                                "supp": adapted.loss(inputs_s, labels_s),
                                "query": adapted.loss(inputs_q, labels_q),
                            },
                        }
                        td_losses.append(task_losses)
                        task_preds_and_labels = {
                            "pre": {
                                "supp": (self.model.predictions(inputs_s), labels_s),
                                "query": (self.model.predictions(inputs_q), labels_q),
                            },
                            "post": {
                                "supp": (adapted.predictions(inputs_s), labels_s),
                                "query": (adapted.predictions(inputs_q), labels_q),
                            },
                        }
                        td_preds_and_labels.append(task_preds_and_labels)
            self.losses.append(td_losses)
            self.preds_and_labels.append(td_preds_and_labels)

    def _build_meta_train_ops(self):
        """Builds meta-learning ops."""
        self.meta_losses = [
            # The default meta-loss is
            tf.reduce_mean([losses["post"]["query"] for losses in td_losses])
            for td_losses in self.losses
        ]
        self.meta_train_op = self.optimizer.minimize(
            tf.reduce_mean(self.meta_losses), var_list=self.model.trainable_parameters
        )

    # --- Abstract methods. ---

    @abc.abstractmethod
    def build_adapted_model(self, task: Task):
        """Builds the adaptation graph. Must be implemented in a subclass."""
