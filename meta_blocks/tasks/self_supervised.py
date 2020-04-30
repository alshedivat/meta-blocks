"""Self-supervised tasks and task distributions for meta-learning.

In the literature, such tasks are also called 'unsupervised'. However,
semantically, these tasks are still supervised, but with labels automatically
generated using various heuristics. Hence, self-supervised is a more accurate
term to use for such tasks.
"""
import logging
from typing import Any, List, Optional, Tuple

import albumentations as alb
import numpy as np
import tensorflow.compat.v1 as tf

from meta_blocks.datasets.base import Dataset, MetaDataset
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
        1.  Sample m points from the underlying dataset (used as a query set),
            which represent m different classes.
        2. Perturb each point k times and form a k-shot support set.

    Parameters
    ----------
    dataset : Dataset

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
        dataset: Dataset,
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
    def support_size(self) -> tf.Tensor:
        return tf.convert_to_tensor(self.num_support_shots * self.num_classes)

    @property
    def support_tensors(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Returns a tuple of support (inputs, labels) tensors.
        The inputs are mapped through dataset's preprocessor.
        """
        return self._support_tensors

    @property
    def query_tensors(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Returns a tuple of query (inputs, labels) tensors.
        The inputs are mapped through dataset's preprocessor.
        """
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
        # TODO: determine if manual device placement is better.
        # with tf.device("CPU:0"):
        inputs = tf.map_fn(
            fn=self.get_augmentation(),
            elems=inputs,
            back_prop=back_prop,
            parallel_iterations=parallel_iterations,
        )
        return inputs, labels

    def _preprocess(
        self,
        inputs: tf.Tensor,
        labels: tf.Tensor,
        back_prop: bool = False,
        parallel_iterations: int = 16,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Applies dataset-provided preprocessor to inputs and labels."""
        if self._preprocessor is not None:
            # TODO: determine if manual device placement is better.
            # with tf.device("CPU:0"):
            inputs = tf.map_fn(
                dtype=tf.float32,  # TODO: look up in self.dataset.
                fn=self._preprocessor,
                elems=(inputs, labels),
                back_prop=back_prop,
                parallel_iterations=parallel_iterations,
            )
        return inputs, labels

    def _build(self):
        """Builds the task internals in the correct name scope."""
        # Input preprocessor.
        self._preprocessor = self.dataset.get_preprocessor()
        # Build input and label tensors by selecting 1 random sample per class.
        indices = [
            tf.random.shuffle(tf.range(tf.shape(x)[0]))[:1]
            for x in self.dataset.data_tensors
        ]
        inputs = [
            tf.gather(x, i, axis=0) for x, i in zip(self.dataset.data_tensors, indices)
        ]
        labels = [tf.fill(tf.shape(x)[:1], k) for k, x in enumerate(inputs)]
        inputs = tf.concat(inputs, axis=0)
        labels = tf.concat(labels, axis=0)
        tensors = self._preprocess(inputs, labels)
        # Build augmented tensors.
        tiles = [[self.num_augmented_shots] + [1] * (len(t.shape) - 1) for t in tensors]
        aug_tensors = self._augment(
            inputs=tf.tile(tensors[0][: self.num_classes], tiles[0]),
            labels=tf.tile(tensors[1][: self.num_classes], tiles[1]),
        )
        # Determine whether support or query are augmented.
        if self.inverse:
            self._support_tensors, self._query_tensors = aug_tensors, tensors
        else:
            self._query_tensors, self._support_tensors = aug_tensors, tensors


class UmtraTaskDistribution(base.TaskDistribution):
    """A distribution that provides access to unsupervised tasks.

    Parameters
    ----------
    meta_dataset : MetaDataset

    num_augmented_shots : int, optional (default: 1)

    inverse : bool, optional (default: True)

    stratified : bool, optional (default: True)

    num_task_batches_to_cache : int, optional (default: 100)

    name: str, optional

    seed : int, optional (default: 42)
    """

    def __init__(
        self,
        meta_dataset: MetaDataset,
        num_augmented_shots: int = 1,
        inverse: bool = True,
        stratified: bool = False,
        num_task_batches_to_cache: int = 100,
        name: Optional[str] = None,
        seed: Optional[int] = 42,
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

        # Setup random number generator.
        self._rng = np.random.RandomState(seed=seed)

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
        self._task_batch = tuple(
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
        for i in range(self.num_task_batches_to_cache):
            requests_batch, feed_list_batch = self.meta_dataset.get_feed_batch(
                # If not stratified, samples classes with replacement,
                # which results in tasks that may have different classes
                # with the same underlying category.
                replace=(not self.stratified)
            )
            self._requests.append(requests_batch)

    def sample_task_feed(self, **_unused_kwargs) -> List[Tuple[tf.Tensor, Any]]:
        """Samples a meta-batch of tasks and returns a feed-dict."""
        if not self._requests:
            self._refresh_requests()
        # Sample a meta-batch of tasks.
        requests_batch = self._requests.pop()
        # Build feed list for the meta-batch of tasks.
        _, feed_list_batch = self.meta_dataset.get_feed_batch(requests=requests_batch)
        return sum(feed_list_batch, [])
