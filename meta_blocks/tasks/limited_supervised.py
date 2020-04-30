"""Supervised tasks and task distributions."""
import logging
from typing import Any, Dict, List, Optional, Tuple

import tensorflow.compat.v1 as tf

from meta_blocks import samplers
from meta_blocks.datasets.base import MetaDataset
from meta_blocks.tasks.supervised import SupervisedTaskDistribution

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()

__all__ = ["LimitedSupervisedTaskDistribution"]


class LimitedSupervisedTaskDistribution(SupervisedTaskDistribution):
    """A distribution that provides access to supervised tasks.

    Pre-samples tasks such that the total number of labeled points upper bounded
    by `max_labeled_points`. If `init_labeled_points` is provided, it samples
    tasks up to `init_labeled_points` at initialization which can be further
    expanded up to `max_labeled_points` by calling `expand` method.

    Parameters
    ----------
    meta_dataset : MetaDataset

    num_query_shots : int, optional (default: 1)

    num_support_shots : int, optional (default: 1)

    max_labeled_points : int, optional

    init_labeled_points : int, optional

    name: str, optional

    seed : int, optional (default: 42)
    """

    def __init__(
        self,
        meta_dataset: MetaDataset,
        sampler_config: Dict[str, Any],
        num_query_shots: int = 1,
        num_support_shots: int = 1,
        max_labeled_points: int = None,
        init_labeled_points: int = None,
        name: Optional[str] = None,
        seed: Optional[int] = 42,
        **_unused_kwargs,
    ):
        super(LimitedSupervisedTaskDistribution, self).__init__(
            meta_dataset=meta_dataset,
            num_query_shots=num_query_shots,
            sampler=samplers.get(**sampler_config),
            name=(name or self.__class__.__name__),
            seed=seed,
        )
        self.num_support_shots = num_support_shots
        self.max_labeled_points = max_labeled_points
        self.init_labeled_points = init_labeled_points

        # Internals.
        self.num_requested_labels = None

    # --- Methods. ---

    def initialize(self, **kwargs):
        """Initializes by pre-sampling supervised tasks."""
        super(LimitedSupervisedTaskDistribution, self).initialize(**kwargs)

        # Determine the initial labeling budget.
        if self.init_labeled_points is None:
            self.init_labeled_points = self.max_labeled_points

        # Sample supervised tasks.
        logger.debug(f"Initializing {self.name}...")
        self.expand(self.init_labeled_points)

    def expand(self, num_labeled_points: int):
        """Expands the number of labeled points by sampling more tasks."""
        # Never exceed the hard budget.
        num_labeled_points = min(num_labeled_points, self.max_labeled_points)

        # Expand the number of requested labels.
        requested_labels_so_far = self.num_requested_labels
        while requested_labels_so_far < num_labeled_points:
            logger.debug(
                f"...requesting more labels: "
                f"{requested_labels_so_far}/{num_labeled_points}"
            )
            # Construct a batch of requests.
            requests_batch, feed_list_batch = self.meta_dataset.get_feed_batch()
            # Get request kwargs.
            request_kwargs_batch = [
                [v for _, v in feed_list[self.num_classes :]]
                for feed_list in feed_list_batch
            ]
            # Sample support labeled ids for the requested tasks.
            support_labeled_ids_batch = self.sampler.select_labeled(
                size=self.support_labels_per_task,
                labels_per_step=self.num_classes,
                feed_dict=dict(sum(feed_list_batch, [])),
            )
            # Save sampled information.
            for i, ids in enumerate(support_labeled_ids_batch):
                requested_labels_so_far += self.query_labels_per_task + len(ids)
                if requested_labels_so_far > num_labeled_points:
                    break
                self._requested_ids.append(ids)
                self._requests.append(requests_batch[i])
                self._requested_kwargs.append(request_kwargs_batch[i])
                self.num_requested_labels = requested_labels_so_far

    def sample_task_feed(self, replace: bool = True) -> List[Tuple[tf.Tensor, Any]]:
        """Samples a meta-batch of tasks and returns a feed list."""
        # Sample a meta-batch of tasks.
        indices_batch = self._rng.choice(
            len(self._requests), size=self.meta_batch_size, replace=replace
        )
        # Build feed list for the meta-batch of tasks.
        requests_batch = [self._requests[i] for i in indices_batch]
        ids_batch = [self._requested_ids[i] for i in indices_batch]
        kwargs_batch = [self._requested_kwargs[i] for i in indices_batch]
        _, feed_list_batch = self.meta_dataset.get_feed_batch(requests=requests_batch)
        task_feed = []
        for i, feed_list in enumerate(feed_list_batch):
            # Truncate feed list.
            task_feed += feed_list[: self.num_classes]
            # Add kwargs feed.
            kwarg_keys = [k for k, _ in feed_list[self.num_classes :]]
            task_feed += [(k, v) for k, v in zip(kwarg_keys, kwargs_batch[i])]
            # Add task-specific feed.
            task_feed += self._task_batch[i].get_feed_list(ids_batch[i])
        return task_feed
