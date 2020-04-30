"""Data management for meta-learning."""
import abc
import logging
from typing import Any, Callable, List, Optional, Tuple

import tensorflow.compat.v1 as tf

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()

__all__ = ["DataSource", "Dataset", "MetaDataset"]


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

    # --- Properties. ---

    @property
    def data_shapes(self):
        """Data shapes after preprocessing. By default same as raw data shapes."""
        return self.raw_data_shapes

    @property
    def data_types(self):
        """Data types after preprocessing. By default same as raw data types."""
        return self.raw_data_types

    # --- Abstract properties. ---

    @property
    @abc.abstractmethod
    def raw_data_shapes(self):
        """Data shapes before preprocessing."""
        raise NotImplementedError("Abstract Property")

    @property
    @abc.abstractmethod
    def raw_data_types(self):
        """Data types before preprocessing."""
        raise NotImplementedError("Abstract Property")

    # --- Methods. ---

    def build(self, **kwargs):
        """Builds the data source in the correct namespace."""
        if not self.built:
            with tf.name_scope(self.name):
                self._build(**kwargs)
            self.built = True
        return self

    def _build(self, **kwargs):
        """Builds the data source. Does nothing by default."""
        pass

    def initialize(self, **kwargs):
        """Initializes the data source. Does nothing by default."""
        pass


class Dataset(abc.ABC):
    """The base abstract class for datasets.

    Internally, `Dataset`s must build data tensors that will contain data at
    execution time. Data tensors can be either `tf.placeholder`s or built around
    `tf.data.Iterator`s. Implementation is typically dataset-specific.

    Parameters
    ----------
    data_shapes : tf.TensorShape

    data_types : tf.DType

    name : str, optional
        The name of the dataset.
    """

    def __init__(
        self,
        data_shapes: tf.TensorShape,
        data_types: tf.DType,
        name: Optional[str] = None,
        **_unused_kwargs,
    ):
        self.data_shapes = data_shapes
        self.data_types = data_types
        self.name = name or self.__class__.__name__

        # Internals.
        self.data_tensors = None
        self.built = False

    # --- Abstract properties. ---

    @property
    @abc.abstractmethod
    def size(self) -> tf.Tensor:
        raise NotImplementedError("Abstract Property")

    # --- Methods. ---

    def build(self):
        """Builds the dataset in the correct namespace."""
        if not self.built:
            with tf.name_scope(self.name):
                self._build()
            self.built = True
        return self

    def get_preprocessor(self) -> Optional[Callable]:
        """Returns a function for data preprocessing or None."""
        pass

    def get_feed_list(self, **kwargs) -> List[Tuple[tf.Tensor, Any]]:
        """Returns a list of (placeholder, value) pairs."""
        return []

    # --- Abstract methods. ---

    @abc.abstractmethod
    def _build(self):
        raise NotImplementedError("Abstract Method")


class ClfDataset(Dataset):
    """The base class for classification datasets.

    Parameters
    ----------
    num_classes : int
        The number of classes.

    name : str, optional
        The name of the dataset.
    """

    def __init__(
        self,
        num_classes: int,
        data_shapes: tf.TensorShape,
        data_types: tf.DType,
        name: Optional[str] = None,
        **_unused_kwargs,
    ):
        super(ClfDataset, self).__init__(
            data_shapes=data_shapes,
            data_types=data_types,
            name=name or self.__class__.__name__,
        )
        self.num_classes = num_classes

    # --- Properties. ---

    def size(self) -> tf.Tensor:
        return tf.reduce_sum([tf.shape(dt)[0] for dt in self.data_tensors])


class MetaDataset(abc.ABC):
    """The base class for meta-datasets."""

    Dataset = Dataset

    def __init__(
        self,
        data_source: DataSource,
        batch_size: int,
        name: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """Instantiates a MetaDataset."""
        self.data_source = data_source
        self.batch_size = batch_size
        self.name = name or self.__class__.__name__
        self.seed = seed

        # Internals.
        self.dataset_batch = tuple(
            self.Dataset(
                data_shapes=self.data_source.raw_data_shapes,
                data_types=self.data_source.raw_data_types,
                name=f"DS{i}",
                **kwargs,
            )
            for i in range(self.batch_size)
        )

        self.built = False

    # --- Methods. ---

    def build(self):
        """Builds internals in the correct name scope."""
        if not self.built:
            logger.debug(f"Building {self.name}...")
            with tf.name_scope(self.name):
                for ds in self.dataset_batch:
                    ds.build()
            self.built = True
        return self

    # --- Abstract methods. ---

    @abc.abstractmethod
    def get_feed_batch(
        self, **kwargs
    ) -> Tuple[Tuple[Any, ...], List[List[Tuple[tf.Tensor, Any]]]]:
        """Returns a list of (placeholder, value) pairs."""
        raise NotImplementedError("Abstract Method")


class ClfMetaDataset(MetaDataset):
    """The base class for meta classification datasets."""

    Dataset = ClfDataset

    def __init__(
        self,
        data_source: DataSource,
        batch_size: int,
        num_classes: int,
        name: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        kwargs["num_classes"] = num_classes
        super(ClfMetaDataset, self).__init__(
            data_source=data_source,
            batch_size=batch_size,
            name=(name or self.__class__.__name__),
            seed=seed,
            **kwargs,
        )
        self.num_classes = num_classes
