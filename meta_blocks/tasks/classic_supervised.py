"""Classical supervised tasks and task distributions."""
import logging
from typing import Any, List, Optional, Tuple

import tensorflow.compat.v1 as tf

from meta_blocks import samplers
from meta_blocks.datasets.base import MetaDataset
from meta_blocks.tasks.supervised import SupervisedTaskDistribution

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()

__all__ = ["ClassicSupervisedTaskDistribution"]


class ClassicSupervisedTaskDistribution(SupervisedTaskDistribution):
    """The classical distribution that provides access to supervised tasks.

    Each time a batch of tasks is requested, it samples brand-new datasets and
    new support and query ids to construct each task in the batch. Essentially,
    it provides access to batches of tasks sampled from a combinatorially large
    (infinite for practical purposes) space of possible tasks. This stream can
    be used for training (classical few-shot learning setting) and evaluation.

    Notes
    -----
    * Using a LimitedSupervisionTaskDistribution is recommended for evaluation
      since it allows to sample tasks from a limited collection which enables
      better reproducibility.

    Parameters
    ----------
    meta_dataset : MetaDataset

    num_query_shots : int, optional (default: 1)

    num_support_shots : int, optional (default: 1)

    num_task_batches_to_cache : int, optional (default: 100)

    name: str, optional

    seed : int, optional (default: 42)
    """

    def __init__(
        self,
        meta_dataset: MetaDataset,
        num_query_shots: int = 1,
        num_support_shots: int = 1,
        num_task_batches_to_cache: int = 100,
        name: Optional[str] = None,
        seed: Optional[int] = 42,
        **_unused_kwargs,
    ):
        super(ClassicSupervisedTaskDistribution, self).__init__(
            meta_dataset=meta_dataset,
            num_query_shots=num_query_shots,
            num_support_shots=num_support_shots,
            sampler=samplers.get(name="uniform", stratified=True),
            name=(name or self.__class__.__name__),
            seed=seed,
        )
        self.num_task_batches_to_cache = num_task_batches_to_cache

    # --- Methods. ---

    def _refresh_requests(self):
        """Re-samples new task requests."""
        for i in range(self.num_task_batches_to_cache):
            # Construct a batch of requests.
            requests_batch, feed_list_batch = self.meta_dataset.get_feed_batch()
            # Get request kwargs.
            # Note: the feed list returned by self.meta_dataset consists of
            #       `num_classes` category id placeholder feeds followed by
            #       the same number of kwargs feed for data preprocessing.
            request_kwargs_batch = [
                [v for _, v in feed_list[self.num_classes :]]
                for feed_list in feed_list_batch
            ]
            # Sample support labeled ids for the requested tasks.
            for k, v in dict(sum(feed_list_batch, [])).items():
                if k is None or v is None:
                    logger.info(k, v)
            support_labeled_ids_batch = self.sampler.select_labeled(
                size=self.support_labels_per_task,
                labels_per_step=self.num_classes,
                feed_dict=dict(sum(feed_list_batch, [])),
            )
            # Save the sampled information.
            self._requests.append(requests_batch)
            self._requested_ids.append(support_labeled_ids_batch)
            self._requested_kwargs.append(request_kwargs_batch)
            self.num_requested_labels += sum(
                len(ids) for ids in support_labeled_ids_batch
            )

    def sample_task_feed(self, **_unused_kwargs) -> List[Tuple[tf.Tensor, Any]]:
        """Samples a meta-batch of tasks and returns a feed list."""
        if not self._requests:
            self._refresh_requests()
        # Get the next batch.
        requests_batch = self._requests.pop()
        ids_batch = self._requested_ids.pop()
        kwargs_batch = self._requested_kwargs.pop()
        _, feed_list_batch = self.meta_dataset.get_feed_batch(requests=requests_batch)
        # Construct task feed.
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
