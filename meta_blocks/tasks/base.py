"""Task interfaces for meta-learning."""

import abc
import logging
from typing import Dict, List, Tuple

import tensorflow.compat.v1 as tf

from meta_blocks import datasets

logger = logging.getLogger(__name__)

__all__ = ["Task", "TaskDistribution"]


class Task(abc.ABC):
    """Abstract base class for tasks."""

    def __init__(self, dataset, num_query_shots=1, name="Task"):
        """Instantiates a Task.

        Args:
            dataset: Dataset
            num_query_shots: int (default: 1)
            name: str (default: "Task")
        """
        self.dataset = dataset
        self.num_query_shots = num_query_shots
        self.name = name

        self.built = False

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes

    @property
    def dataset_size(self) -> tf.Tensor:
        return self.dataset.size

    @property
    def query_size(self) -> int:
        return self.num_query_shots * self.num_classes

    @property
    @abc.abstractmethod
    def support_size(self) -> tf.Tensor:
        raise NotImplementedError("Abstract Method")

    def build(self):
        """Builds the task internals in the correct name scope."""
        if not self.built:
            with tf.name_scope(self.name):
                self._build()
            self.built = True
        return self

    @abc.abstractmethod
    def _build(self):
        raise NotImplementedError("Abstract Method")

    @property
    @abc.abstractmethod
    def support_tensors(self):
        raise NotImplementedError("Abstract Method")

    @property
    @abc.abstractmethod
    def query_tensors(self):
        raise NotImplementedError("Abstract Method")

    def get_feed_list(self, **kwargs) -> List[Dict[tf.Tensor, str]]:
        """Creates a feed list needed for the task."""
        return []


class TaskDistribution(abc.ABC):
    """
    Base abstract class for task distributions.

    Task distributions must keep track of which tasks have been used for
    training already and which to use next.
    """

    def __init__(
        self,
        meta_dataset: datasets.MetaDataset,
        num_query_shots: int = 1,
        name: str = "TaskDistribution",
    ):
        self.meta_dataset = meta_dataset
        self.num_query_shots = num_query_shots
        self.name = name

        # Internals.
        self._task_batch = None

        self.built = False

    @property
    def meta_batch_size(self):
        return self.meta_dataset.batch_size

    @property
    def num_classes(self):
        return self.meta_dataset.num_classes

    @property
    def task_batch(self) -> Tuple[Task]:
        return self._task_batch

    def build(self):
        if not self.built:
            with tf.name_scope(self.name):
                self._build()
            self.built = True
        return self

    @abc.abstractmethod
    def _build(self):
        raise NotImplementedError("Abstract Method")

    @abc.abstractmethod
    def initialize(self, **kwargs):
        raise NotImplementedError("Abstract Method")

    @abc.abstractmethod
    def expand(self, **kwargs):
        raise NotImplementedError("Abstract Method")

    @abc.abstractmethod
    def sample_task_feed(self, **kwargs) -> Dict[tf.Tensor, str]:
        raise NotImplementedError("Abstract Method")
