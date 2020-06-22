"""Classical supervised tasks and task distributions."""
import logging
from typing import Iterator, Optional

import tensorflow.compat.v1 as tf

from meta_blocks import samplers
from meta_blocks.datasets.base import ClfMetaDataset, FeedList
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
    meta_dataset : ClfMetaDataset

    num_query_shots : int, optional (default: 1)

    num_support_shots : int, optional (default: 1)

    num_task_batches_to_cache : int, optional (default: 100)

    name: str, optional

    seed : int, optional (default: 42)
    """

    def __init__(
        self,
        meta_dataset: ClfMetaDataset,
        num_query_shots: int = 1,
        num_support_shots: int = 1,
        num_batches_per_epoch: int = 100,
        name: Optional[str] = None,
        stratified: bool = True,
        **_unused_kwargs,
    ):
        super(ClassicSupervisedTaskDistribution, self).__init__(
            meta_dataset=meta_dataset,
            num_query_shots=num_query_shots,
            num_support_shots=num_support_shots,
            name=(name or self.__class__.__name__),
        )
        self.num_batches_per_epoch = num_batches_per_epoch
        self.stratified = stratified

    # --- Methods. ---

    def initialize(self, **_unused_kwargs):
        """Initializes the task distribution using uniform sampler."""
        sampler = samplers.get(name="uniform", stratified=self.stratified)
        super(ClassicSupervisedTaskDistribution, self).initialize(
            sampler=sampler.build(task_dist=self)
        )

    def _refresh_requests(self):
        """Re-samples new task requests."""
        logger.debug(f"Sampling new task batches from {self.name}... ")
        for i in range(self.num_batches_per_epoch):
            # Construct a batch of requests.
            requests_batch, feed_list = self.meta_dataset.request_datasets(
                unique_classes=True
            )
            support_labeled_ids_batch = self.sampler.select_labeled(
                size=self.support_labels_per_task, feed_dict=dict(feed_list)
            )
            # Save the sampled information.
            self._requests.append(requests_batch)
            self._requested_ids.append(support_labeled_ids_batch)
            self.num_requested_labels += sum(
                len(ids) for ids in support_labeled_ids_batch
            )

    def sample_task_feed(self, **_unused_kwargs) -> FeedList:
        """Samples a meta-batch of tasks and returns a feed list."""
        if not self._requests:
            self._refresh_requests()
        # Get the next batch.
        requests_batch = self._requests.pop()
        ids_batch = self._requested_ids.pop()
        _, feed_list = self.meta_dataset.request_datasets(requests_batch)
        # Construct task feed.
        for task, ids in zip(self.task_batch, ids_batch):
            feed_list.extend(task.get_feed_list(ids))
        return feed_list

    def epoch(self, **kwargs) -> Iterator[FeedList]:
        """A generator that yields task batches."""
        for i in range(self.num_batches_per_epoch):
            yield self.sample_task_feed()
