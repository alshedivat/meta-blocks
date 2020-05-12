"""Data management for meta-learning."""
import abc
import logging
from typing import Any, List, Optional, Tuple

import tensorflow.compat.v1 as tf

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()

__all__ = ["DataSource", "Dataset", "MetaDataset"]


# Types.
FeedList = List[Tuple[tf.Tensor, Any]]
DatasetRequest = Any


class DataSource(abc.ABC):
    """The base abstract class for data sources.

    Data sources are the underlying data structures that provide access to the
    actual data from which meta-datasets can further sample small datasets.
    Each dataset must implement a data source.

    Parameters
    ----------
    data_dir : str
        Path to the directory that contains the data.

    name : str, optional
        The name of the dataset.
    """

    def __init__(self, data_dir: str, name: Optional[str] = None):
        self.data_dir = data_dir
        self.name = name or self.__class__.__name__

        # Internals.
        self.built = False

    # --- Abstract properties. ---

    @property
    @abc.abstractmethod
    def data_shapes(self):
        """Data shapes before preprocessing."""

    @property
    @abc.abstractmethod
    def data_types(self):
        """Data types before preprocessing."""

    # --- Methods. ---

    def build(self, **kwargs):
        """Builds the data source in the correct namespace."""
        if not self.built:
            with tf.name_scope(self.name):
                self._build(**kwargs)
            self.built = True
        return self

    # --- Abstract methods. ---

    @abc.abstractmethod
    def _build(self, **kwargs):
        """Builds the data source. Must be implemented by a subclass."""


class Dataset(abc.ABC):
    """The base abstract class for datasets.

    Internally, `Dataset` must build data tensors that will contain data at
    execution time. Implementation is typically dataset-specific.

    Parameters
    ----------
    name : str, optional
        The name of the dataset.
    """

    def __init__(self, name: Optional[str] = None, **_unused_kwargs):
        self.name = name or self.__class__.__name__

        # Internals.
        self.data_tensors = None
        self.built = False

    # --- Abstract properties. ---

    @property
    @abc.abstractmethod
    def size(self) -> tf.Tensor:
        """Returns the dynamic dataset size. Must be implemented by a subclass."""

    # --- Methods. ---

    def build(self):
        """Builds the dataset in the correct namespace."""
        if not self.built:
            with tf.name_scope(self.name):
                self._build()
            self.built = True
        return self

    # --- Abstract methods. ---

    @abc.abstractmethod
    def _build(self):
        """Builds dataset internals. Must be implemented by a subclass."""


class ClfDataset(Dataset):
    """The base class for classification datasets.

    Parameters
    ----------
    num_classes : int
        The number of classes.

    name : str, optional
        The name of the dataset.
    """

    def __init__(self, num_classes: int, name: Optional[str] = None, **_unused_kwargs):
        super(ClfDataset, self).__init__(name=name or self.__class__.__name__)
        self.num_classes = num_classes

        # Internals.
        self._size = None

    # --- Properties. ---

    @property
    def size(self) -> int:
        """Returns the size of the dataset."""
        return self._size


class MetaDataset(abc.ABC):
    """The base class for meta-datasets.

    Parameters
    ----------
    batch_size : int
        The size of the meta-batch.

    data_sources : list of DataSources
        The sources of data used for building datasets.

    num_classes : int
        The number of classes.

    name : str, optional
        The name of the dataset.
    """

    def __init__(
        self,
        batch_size: int,
        data_sources: List[DataSource],
        name: Optional[str] = None,
    ):
        self.batch_size = batch_size
        self.data_sources = data_sources
        self.name = name or self.__class__.__name__

        # Internals.
        self.built = False

    # --- Methods. ---

    def build(self):
        """Builds internals in the correct name scope."""
        if not self.built:
            logger.debug(f"Building {self.name}...")
            with tf.name_scope(self.name):
                self._build()
            self.built = True
        return self

    # --- Abstract methods. ---

    @abc.abstractmethod
    def _build(self):
        """Builds meta-dataset internals. Must be implemented in a subclass."""

    @abc.abstractmethod
    def request_datasets(
        self, requests_batch: Optional[Any] = None, unique_classes: bool = True
    ) -> Tuple[Tuple[DatasetRequest], FeedList]:
        """Returns a batch of dataset requests and the corresponding feed."""


class ClfMetaDataset(MetaDataset):
    """The base class for meta classification datasets.

    Parameters
    ----------
    batch_size : int
        The size of the meta-batch.

    num_classes : int
        The number of classes of the provided datasets.

    data_sources : list of DataSources
        The sources of data used for building datasets.

    name : str, optional
        The name of the dataset.
    """

    def __init__(
        self,
        batch_size: int,
        num_classes: int,
        data_sources: List[DataSource],
        name: Optional[str] = None,
    ):
        super(ClfMetaDataset, self).__init__(
            batch_size=batch_size,
            data_sources=data_sources,
            name=(name or self.__class__.__name__),
        )
        self.num_classes = num_classes
