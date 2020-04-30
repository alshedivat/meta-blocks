"""Task interfaces for meta-learning."""
import abc
import logging
from typing import Any, Dict, List, Optional, Tuple

import tensorflow.compat.v1 as tf

from meta_blocks.datasets.base import Dataset, MetaDataset

logger = logging.getLogger(__name__)

__all__ = ["Task", "TaskDistribution"]


class Task(abc.ABC):
    """Abstract base class for tasks.

    Tasks wrap around datasets and must implement an interface for data access.
    Currently, tasks are designed to only support few-shot classification.
    Support for other types of tasks (imitation, RL, etc.) is planned.

    Each task must provide access to a balanced query set with `num_query_shots`
    points per class. The support set can be arbitrary, and it is up to the
    subclass to determine how its support set is defined and constructed.

    Parameters
    ----------
    dataset : Dataset

    num_query_shots : int, optional (default: 1)

    name: str, optional
    """

    def __init__(
        self,
        dataset: Dataset,
        num_query_shots: Optional[int] = 1,
        name: Optional[str] = None,
    ):
        self.dataset = dataset
        self.num_query_shots = num_query_shots
        self.name = name or self.__class__.__name__

        self.built = False

    # --- Properties. ---

    @property
    def num_classes(self) -> tf.Tensor:
        return self.dataset.num_classes

    @property
    def dataset_size(self) -> tf.Tensor:
        return self.dataset.size

    @property
    def query_size(self) -> tf.Tensor:
        return self.num_query_shots * self.num_classes

    # --- Abstract properties. ---

    @property
    @abc.abstractmethod
    def support_size(self) -> tf.Tensor:
        """Returns size of the support set. Must be implemented by a subclass."""
        raise NotImplementedError("Abstract Method")

    @property
    @abc.abstractmethod
    def support_tensors(self) -> Tuple[tf.Tensor, ...]:
        """Returns a tuple of support tensors (each corresponds to a class).
        Must be implemented by a subclass.
        """
        raise NotImplementedError("Abstract Method")

    @property
    @abc.abstractmethod
    def query_tensors(self) -> Tuple[tf.Tensor, ...]:
        """Returns a tuple of query tensors (each corresponds to a class).
        Must be implemented by a subclass.
        """
        raise NotImplementedError("Abstract Method")

    # --- Methods. ---

    def build(self):
        """Builds the task internals in the correct name scope."""
        if not self.built:
            with tf.name_scope(self.name):
                self._build()
            self.built = True
        else:
            logger.warning(f"{self.name} is already built!")
        return self

    def get_feed_list(self, **kwargs) -> List[Dict[tf.Tensor, Any]]:
        """Creates a feed list for any placeholders created by the task."""
        return []

    # --- Abstract methods. ---

    @abc.abstractmethod
    def _build(self) -> None:
        """Builds internals of the task. Must be implemented by a subclass."""
        raise NotImplementedError("Abstract Method")


class TaskDistribution(abc.ABC):
    """
    Base abstract class for task distributions.

    Task distributions wrap around meta-datasets and must implement an interface
    for selecting/sampling (batches of) tasks (and the corresponding datasets)
    used for meta-training or evaluation.

    Notes
    -----
    * Task distributions are currently implemented with graph execution mode in
      mind, i.e., they build static graphs for `meta_dataset.batch_size` tasks.
      The data provided by each task depends on some internal placeholders, e.g.
      the ids of the categories the data is pulled from for each class and ids
      of the instances used to construct support and query sets. After building
      a task distribution, we get access to a symbolic `task_batch` upon which
      we can further build other parts of the graph (adaptation, models, etc.).

    * At meta-training time, at each step, we can call `sample_task_feed` which
      returns feed values for all necessary placeholders that must be provided
      to the `tf.Session.run` method.

    Parameters
    ----------
    meta_dataset : MetaDataset

    num_query_shots : int, optional (default: 1)

    name: str, optional
    """

    def __init__(
        self,
        meta_dataset: MetaDataset,
        num_query_shots: Optional[int] = 1,
        name: Optional[str] = None,
    ):
        self.meta_dataset = meta_dataset
        self.num_query_shots = num_query_shots
        self.name = name or self.__class__.__name__

        # Internals.
        self._task_batch = None

        self.built = False

    # --- Properties. ---

    @property
    def meta_batch_size(self) -> int:
        """Returns the number of tasks in the meta-batch."""
        return self.meta_dataset.batch_size

    @property
    def num_classes(self) -> int:
        """Returns the number of classes in each task."""
        return self.meta_dataset.num_classes

    @property
    def task_batch(self) -> Tuple[Task, ...]:
        """Returns a tuple of tasks from the distribution."""
        return self._task_batch

    # --- Methods. ---

    def build(self):
        if not self.built:
            with tf.name_scope(self.name):
                self._build()
            self.built = True
        else:
            logger.warning(f"{self.name} is already built!")
        return self

    # --- Abstract methods. ---

    @abc.abstractmethod
    def _build(self) -> None:
        raise NotImplementedError("Abstract Method")

    @abc.abstractmethod
    def initialize(self, **kwargs) -> None:
        raise NotImplementedError("Abstract Method")

    @abc.abstractmethod
    def sample_task_feed(self, **kwargs) -> List[Tuple[tf.Tensor, Any]]:
        raise NotImplementedError("Abstract Method")
