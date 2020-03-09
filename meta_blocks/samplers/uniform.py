"""Uniform sampling."""

import tensorflow.compat.v1 as tf

from typing import Optional, Tuple

from meta_blocks.tasks import SupervisedTask
from meta_blocks.samplers import base

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()

__all__ = ["UniformSampler"]


class UniformSampler(base.Sampler):
    """Samples instances uniformly at random."""

    stateful = False

    def __init__(self, stratified=False, name="UniformSampler", **kwargs):
        super(UniformSampler, self).__init__(stratified=stratified, name=name)

        # Internal.
        self._size = None
        self._selected_indices = None

    def _build(self, tasks: Tuple[SupervisedTask], **kwargs):
        """Builds a tuple of selected indices tensors."""
        del kwargs  # Unused.
        self._size = tf.placeholder(tf.int32, shape=(), name="size")

        # Build selected indices tensors.
        selected_indices = []
        for i, task in enumerate(tasks):
            # Compute scores.
            indices, scores = self.compute_scores(task)
            # Select indices of the elements to be labeled.
            if self.stratified:
                task_indices = self.select_indices_stratified(
                    size=self._size,
                    scores=scores,
                    clusters=task._support_labels_raw,
                    indices=indices,
                )
            else:
                task_indices = self.select_indices(
                    size=self._size, indices=indices, scores=scores
                )
            selected_indices.append(task_indices)
        self._selected_indices = tuple(selected_indices)

    def compute_scores(self, task: SupervisedTask, **kwargs):
        del kwargs  # Unused.
        indices = tf.range(task.unlabeled_support_size, dtype=tf.int32)
        scores = tf.random.uniform(shape=(task.unlabeled_support_size,))
        return indices, scores

    def select_labeled(
        self,
        size: int,
        sess: Optional[tf.Session] = None,
        feed_dict: Optional[dict] = None,
        **kwargs
    ) -> Tuple[tf.Tensor]:
        """Return an actively selected labeled data points from the dataset."""
        del kwargs  # Unused.
        if sess is None:
            sess = tf.get_default_session()
        if feed_dict is None:
            feed_dict = {}
        feed_dict[self._size] = size
        return sess.run(tuple(self._selected_indices), feed_dict=feed_dict)
