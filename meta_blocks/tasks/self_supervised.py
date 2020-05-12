"""Self-supervised tasks and task distributions for meta-learning.

In the literature, such tasks are also called 'unsupervised'. However,
semantically, these tasks are still supervised, but with labels automatically
generated using various heuristics. Hence, self-supervised is a more accurate
term to use for such tasks.
"""
import logging
from typing import Optional, Tuple

import albumentations as alb
import numpy as np
import tensorflow.compat.v1 as tf

from meta_blocks.datasets.base import ClfDataset, ClfMetaDataset, FeedList
from meta_blocks.tasks import base

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()

__all__ = ["UmtraTask", "UmtraTaskDistribution"]


class UmtraTask(base.Task):
    """A self-supervised meta-learning task.

    The tasks are constructed through data augmentation using a variation of
    the method proposed in [1]:

    1. Sample m points from the underlying dataset (used as a query set),
       which represent m different classes.
    2. Perturb each point k times and form a k-shot support set.

    Parameters
    ----------
    dataset : ClfDataset

    num_augmented_shots: int (default: 1)

    inverse: bool (default: False)

    stratified: bool (default: False)

    name: str (default: "UmtraTask")

    References
    ----------
    .. [1]: Khodadadeh, Bölöni, Shah. "Unsupervised Meta-Learning for Few-Shot
            Image Classification." NeurIPS, 2019.
    """

    def __init__(
        self,
        dataset: ClfDataset,
        num_augmented_shots: int = 1,
        inverse: bool = True,
        name: Optional[str] = None,
    ):
        super(UmtraTask, self).__init__(
            dataset=dataset, num_query_shots=None, name=name
        )
        self.inverse = inverse

        # Determine the number of support and query shots.
        self.num_augmented_shots = num_augmented_shots
        if inverse:
            self.num_support_shots = num_augmented_shots
            self.num_query_shots = 1
        else:
            self.num_support_shots = 1
            self.num_query_shots = num_augmented_shots

        # Internals.
        self._preprocessor = None
        self._support_tensors = None
        self._query_tensors = None

    # --- Properties. ---

    @property
    def num_ways(self) -> int:
        """Returns the number of classification ways."""
        return self.dataset.num_classes

    @property
    def query_size(self) -> int:
        """Returns the size of the query set."""
        return self.num_query_shots * self.num_ways

    @property
    def support_size(self) -> int:
        """Returns the size of the support set."""
        return self.num_support_shots * self.num_ways

    @property
    def support_tensors(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Returns a tuple of support (inputs, labels) tensors."""
        return self._support_tensors

    @property
    def query_tensors(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Returns a tuple of query (inputs, labels) tensors."""
        return self._query_tensors

    # --- Methods. ---

    @staticmethod
    def get_augmentation(
        aug_prob: float = 1.0,
        rotate_limit: int = 45,
        scale_limit: float = 0.1,
        shift_limit: float = 0.3,
    ):
        """Returns image augmentor."""
        aug = alb.Compose(
            [
                # Flips, shifts, scales, rotations.
                alb.ShiftScaleRotate(
                    shift_limit=shift_limit,
                    scale_limit=scale_limit,
                    rotate_limit=rotate_limit,
                    p=aug_prob,
                ),
                # Transforms.
                alb.ElasticTransform(),
            ],
            p=aug_prob,
        )

        def augment(image):
            augmented = aug(image=image.numpy())
            return augmented["image"]

        def _mapper(image):
            image_aug = tf.py_function(augment, [image], image.dtype)
            image_aug.set_shape(image.shape)
            return image_aug

        return _mapper

    def _augment(
        self,
        inputs: tf.Tensor,
        labels: tf.Tensor,
        back_prop: bool = False,
        parallel_iterations: int = 16,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Applies augmentation to the inputs and labels."""
        inputs = tf.map_fn(
            fn=self.get_augmentation(),
            elems=inputs,
            back_prop=back_prop,
            parallel_iterations=parallel_iterations,
        )
        return inputs, labels

    def _build(self):
        """Builds the task internals in the correct name scope."""
        # Build input and label tensors by selecting 1 random sample per class.
        indices = [
            tf.random.shuffle(tf.range(tf.shape(x)[0]))[:1]
            for x in self.dataset.data_tensors
        ]
        inputs = [
            tf.gather(x, i, axis=0) for x, i in zip(self.dataset.data_tensors, indices)
        ]
        labels = [tf.fill(tf.shape(x)[:1], k) for k, x in enumerate(inputs)]
        tensors = tf.concat(inputs, axis=0), tf.concat(labels, axis=0)
        # Build augmented tensors.
        tiles = [[self.num_augmented_shots] + [1] * (len(t.shape) - 1) for t in tensors]
        aug_tensors = self._augment(
            inputs=tf.tile(tensors[0][: self.num_ways], tiles[0]),
            labels=tf.tile(tensors[1][: self.num_ways], tiles[1]),
        )
        # Determine whether support or query are augmented.
        if self.inverse:
            self._support_tensors, self._query_tensors = aug_tensors, tensors
        else:
            self._query_tensors, self._support_tensors = aug_tensors, tensors


class UmtraTaskDistribution(base.TaskDistribution):
    """A distribution that provides access to self-supervised UMTRA tasks.

    Parameters
    ----------
    meta_dataset : ClfMetaDataset

    num_augmented_shots : int, optional (default: 1)

    inverse : bool, optional (default: True)

    stratified : bool, optional (default: True)

    num_task_batches_to_cache : int, optional (default: 100)

    name: str, optional

    seed : int, optional (default: 42)
    """

    def __init__(
        self,
        meta_dataset: ClfMetaDataset,
        num_augmented_shots: int = 1,
        inverse: bool = True,
        stratified: bool = False,
        num_task_batches_to_cache: int = 100,
        name: Optional[str] = None,
        **_unused_kwargs,
    ):
        super(UmtraTaskDistribution, self).__init__(
            meta_dataset=meta_dataset, num_query_shots=None, name=name
        )
        self.inverse = inverse
        self.stratified = stratified
        self.num_task_batches_to_cache = num_task_batches_to_cache

        # Determine the number of support and query shots.
        self.num_augmented_shots = num_augmented_shots
        if inverse:
            self.num_support_shots = num_augmented_shots
            self.num_query_shots = 1
        else:
            self.num_support_shots = 1
            self.num_query_shots = num_augmented_shots

        # Random states must be seeded globally.
        self._rng = np.random

        # Internals.
        self.num_requested_labels = None
        self._requests = None

    # --- Properties. ---

    @property
    def query_labels_per_task(self) -> int:
        return self.num_classes * self.num_query_shots

    @property
    def support_labels_per_task(self) -> int:
        return self.num_classes * self.num_support_shots

    # --- Methods. ---

    def _build(self):
        # Build a batch of tasks.
        self.task_batch = tuple(
            UmtraTask(
                dataset=dataset,
                inverse=self.inverse,
                num_augmented_shots=self.num_augmented_shots,
                name=f"UmtraTask{i}",
            ).build()
            for i, dataset in enumerate(self.meta_dataset.dataset_batch)
        )

    def initialize(self, **_unused_kwargs):
        self._requests = []
        self.num_requested_labels = 0

    def _refresh_requests(self):
        """Expands the number of labeled points by sampling more tasks."""
        logger.debug(f"Sampling new task batches from {self.name}... ")
        for i in range(self.num_task_batches_to_cache):
            requests_batch, _ = self.meta_dataset.request_datasets(
                # If not stratified, samples classes with replacement,
                # which results in tasks that may have different classes
                # with the same underlying category.
                unique_classes=self.stratified
            )
            self._requests.append(requests_batch)

    def sample_task_feed(self, **_unused_kwargs) -> FeedList:
        """Samples a meta-batch of tasks and returns a feed-dict."""
        if not self._requests:
            self._refresh_requests()
        # Sample a meta-batch of tasks.
        requests_batch = self._requests.pop()
        # Build feed list for the meta-batch of tasks.
        _, feed_list = self.meta_dataset.request_datasets(requests_batch)
        return feed_list
