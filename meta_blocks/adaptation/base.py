"""Base classes and functionality for model adaptation."""

import abc
import collections
import logging

import tensorflow.compat.v1 as tf

from .. import common

__all__ = ["AdaptationStrategy"]

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


class AdaptationStrategy(abc.ABC):
    """Base base class for meta-trainable adaptation strategies.

    Parameters
    ----------
    model : object
        The model being adapted.

    optimizer : object
        The optimizer to use for meta-training.

    tasks : tuple of Tasks
        A tuple of tasks that provide access to data.

    mode : str
        The description string.

    name : str
        The description string.

    \*\*kwargs : dict, optional
        Additional arguments
    """

    def __init__(
            self,
            model,
            optimizer,
            tasks,
            mode=common.ModeKeys.TRAIN,
            name="AdaptationStrategy",
            **kwargs,
    ):

        # Instantiates an AdaptationStrategy.

        self.model = model
        self.optimizer = optimizer
        self.tasks = tasks

        # Internals.
        self.loss = None
        self.train_op = None
        self.preds_and_labels = None

        # Misc.
        self.mode = mode
        self.name = name
        self.built = False

    @abc.abstractmethod
    def _build_adaptation(self):
        raise NotImplementedError("Abstract Method")

    @abc.abstractmethod
    def _build_meta_losses(self):
        """Builds the the meta-loss."""
        raise NotImplementedError("Abstract Method")

    @abc.abstractmethod
    def _build_meta_eval(self):
        """Builds predictions and labels for the query set."""
        raise NotImplementedError("Abstract Method")

    def _build_meta_train(self):
        """Builds meta-update op."""
        meta_grads = collections.defaultdict(list)
        with tf.name_scope("meta-learning"):
            # Build meta-loss.
            losses = self._build_meta_losses()
            meta_loss = tf.reduce_mean(losses)

            # Build meta-gradients.
            for loss in losses:
                grads_and_vars = self.optimizer.compute_gradients(
                    loss, self.model.parameters
                )
                for g, v in grads_and_vars:
                    if g is None:
                        continue
                    meta_grads[v].append(g)

            # Build meta-train op.
            meta_train_op = self.optimizer.apply_gradients(
                [(tf.reduce_mean(g, axis=0), v) for v, g in meta_grads.items()]
            )

            return meta_loss, meta_train_op

    def build(self):
        """Builds the graph for adaptation strategy."""
        logger.info(f"Building {self.name}...")

        # Build adaptation.
        logger.info("Building adaptation graph...")
        with tf.name_scope("adaptation"):
            self._build_adaptation()

        if self.mode == common.ModeKeys.TRAIN:
            # Build meta-update op.
            logger.debug("Building meta-training op...")
            with tf.name_scope("meta-train"):
                self.loss, self.train_op = self._build_meta_train()
        else:  # self.mode in {common.ModeKeys.EVAL, common.ModeKeys.PREDICT}
            # Build query predictions and labels.
            logger.debug("Building predictions and labels...")
            with tf.name_scope("meta-eval"):
                self.preds_and_labels = self._build_meta_eval()

        logger.debug("Done.")

    @abc.abstractmethod
    def get_adapted_model(self, **kwargs):
        """Should return adapted model."""
        raise NotImplementedError("Abstract Method")

    def get_feed_list(self, **kwargs):
        """No feed dict by default."""
        return []
