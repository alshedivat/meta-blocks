"""Dataset management."""

import abc
import logging

import tensorflow.compat.v1 as tf

from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()

__all__ = ["Category", "DataPool", "Dataset", "MetaDataset"]


class Category(abc.ABC):
    """An abstract class for representing different categories of instances.
    Provides access to the underlying data through the tf.data API.

    Parameters
    ----------
    data_dir : str
        The data directory.

    name : str
        The description string.

    """

    def __init__(self, data_dir: str, name: str) -> None:
        """Instantiates a Category."""
        self.data_dir = data_dir
        self.name = name

        # Internals.
        self._size = None
        self._dataset = None
        self._iterator = None
        self._string_handle = None

        self.built = False

    @property
    def size(self) -> int:
        return self._size

    @property
    def dataset(self) -> tf.data.Dataset:
        return self._dataset

    @property
    def iterator(self) -> tf.data.Iterator:
        return self._iterator

    @property
    def string_handle(self) -> tf.Tensor:
        return self._string_handle

    def build(self, *args, **kwargs):
        if not self.built:
            self._build(*args, **kwargs)
        return self

    @abc.abstractmethod
    def _build(self, *args, **kwargs):
        """Internal fucntion for building tf.datasets.Dataset for the
        underlying datasets resources. Must be implemented by a subclass.
        """
        raise NotImplementedError("Abstract Method")


class DataPool(object):
    """Represents a pool of data.

    Parameters
    ----------
    categories : The description string.
        The description string.

    name : str, optional (default='DataPool')
        The description string.


    Attributes
    ----------
    num_categories : Type and default value.
        The description string.

    category_handles : Type and default value.
        The description string.

    output_shapes : Type and default value.
        The description string.
    """

    def __init__(self, categories: List[Category], name: str = "DataPool"):
        """Instantiates a DataPool."""
        self.categories = categories
        self.name = name

        # Internals.
        self._category_handles = None
        self.built = False

    @property
    def num_categories(self) -> int:
        return len(self.categories)

    @property
    def category_handles(self) -> List[str]:
        return self._category_handles

    @property
    def output_types(self):
        return tf.data.get_output_types(self.categories[0].dataset)

    @property
    def output_shapes(self):
        return tf.data.get_output_shapes(self.categories[0].dataset)

    def build(self, **kwargs):
        """Builds all managed data categories."""
        if not self.built:
            logger.info(f"Building {self.name}...")
            for c in self.categories:
                c.build(**kwargs)
            self.built = True
        return self

    def initialize(self, sess):
        """Initializes all managed data categories within the given session."""
        logger.info(f"Initializing {self.name}...")
        self._category_handles = sess.run(
            [c.string_handle for c in self.categories]
        )
        return self


class Dataset(object):
    """Generates dataset tensors by pulling data from specified categories.

    Parameters
    ----------
    num_classes : int
        The number of classes.

    name : str, optional (default="Dataset")
        The name of dataset.

    kwargs

    Attributes
    ----------
    data_tensors : Type and default value.
        The description string.

    size : Type and default value.
        The description string.
    """

    def __init__(self, num_classes: int, name: str = "Dataset", **kwargs):
        self.num_classes = num_classes
        self.name = name

        # Internals.
        self._data_tensors = None
        self._string_handle_phs = None

        self.built = False

    @property
    def data_tensors(self) -> Tuple[tf.Tensor]:
        return self._data_tensors

    @property
    def size(self) -> tf.Tensor:
        return tf.reduce_sum([tf.shape(dt)[0] for dt in self.data_tensors])

    def build(self, output_types, output_shapes):
        """The description string.

        Parameters
        ----------
        output_types : Type and default value.
            The description string.

        output_shapes : Type and default value.
            The description string.

        Returns
        -------
        self : object
        """
        if not self.built:
            data_tensors = []
            string_handle_phs = []
            with tf.name_scope(self.name):
                for k in range(self.num_classes):
                    string_handle_ph = tf.placeholder(
                        tf.string, shape=[], name=f"iterator_handle_class_{k}"
                    )
                    iterator = tf.data.Iterator.from_string_handle(
                        string_handle_ph,
                        output_types=output_types,
                        output_shapes=output_shapes,
                    )
                    data_tensor = iterator.get_next()
                    data_tensors.append(data_tensor)
                    string_handle_phs.append(string_handle_ph)
            self._data_tensors = tuple(data_tensors)
            self._string_handle_phs = tuple(string_handle_phs)
            self.built = True
        return self

    def get_preprocessor(self) -> Optional[Callable]:
        """Returns a function for input preprocessing or None."""
        pass

    def get_feed_list(
        self, string_handles: Tuple[str]
    ) -> List[Tuple[tf.Tensor, str]]:
        """Returns tensors with data and a feed dict with dependencies."""
        if len(string_handles) != self.num_classes:
            raise ValueError("Incorrect number of string handles provided.")
        return list(zip(self._string_handle_phs, string_handles))


class MetaDataset(object):
    """Generates datasets from the provided pool of data."""

    Dataset = Dataset

    def __init__(
        self,
        data_pool: DataPool,
        num_classes: int,
        batch_size: int,
        name: str = "MetaDataset",
        **kwargs,
    ):
        """Instantiates a MetaDataset."""
        self.data_pool = data_pool
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.name = name

        # Internals.
        self._dataset_batch = tuple(
            self.Dataset(self.num_classes, name=f"DS_{i}", **kwargs)
            for i in range(self.batch_size)
        )

        self.built = False

    @property
    def num_categories(self):
        return self.data_pool.num_categories

    @property
    def dataset_batch(self):
        return self._dataset_batch

    def build(self):
        if not self.built:
            logger.info(f"Building {self.name}...")
            with tf.name_scope(self.name):
                for ds in self._dataset_batch:
                    ds.build(
                        output_types=self.data_pool.output_types,
                        output_shapes=self.data_pool.output_shapes,
                    )
            self.built = True
        return self

    def get_feed_list_batch(
        self, requests: Tuple[Tuple[int]]
    ) -> List[List[Tuple[tf.Tensor, str]]]:
        """Returns a feed list for the requested meta-batch of datasets."""
        if len(requests) > self.batch_size:
            raise ValueError("The dataset request is incompatible.")
        # Get feed dicts for each request.
        feed_lists = []
        for n, category_ids in enumerate(requests):
            feed_lists.append(
                self._dataset_batch[n].get_feed_list(
                    [self.data_pool.category_handles[i] for i in category_ids]
                )
            )
        return feed_lists
