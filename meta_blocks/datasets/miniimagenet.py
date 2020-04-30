"""The Mini-ImageNet dataset."""

import glob
import logging
import os
import random
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import tensorflow.compat.v1 as tf

from meta_blocks.datasets import base

logger = logging.getLogger(__name__)

__all__ = [
    "MiniImageNetCategory",
    "MiniImageNetDataSource",
    "MiniImageNetDataset",
    "MiniImageNetMetaDataset",
]


def _bytes_feature_list(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


class MiniImageNetCategory(base.DataSource):
    """Represents data source for a single mini-ImageNet category.

    Parameters
    ----------
    data_dir : str
        Path to the directory that contains the character data.

    rotation : int (default: 0)
        Rotation of the character in degrees.

    name : str, optional
        The name of the dataset.
    """

    RAW_SHAPE = (None,)
    RAW_DTYPE = tf.string
    RAW_IMG_SHAPE = (180, 180, 3)
    IMG_SHAPE = (84, 84, 3)
    IMG_DTYPE = tf.float32

    def __init__(
        self,
        data_dir: str,
        cache: bool = True,
        max_size: Optional[int] = None,
        shuffle: bool = True,
        name: Optional[str] = None,
    ):
        super(MiniImageNetCategory, self).__init__(
            data_dir, name=(name or self.__class__.__name__)
        )
        self.cache = cache
        self.max_size = max_size
        self.shuffle = shuffle

        # Internals.
        self.dataset = None
        self.iterator = None
        self.string_handle = None
        self.size = None

    # --- Properties. ---

    @property
    def raw_data_shapes(self):
        return self.RAW_SHAPE

    @property
    def raw_data_types(self):
        return self.RAW_DTYPE

    # --- Class methods. ---

    @classmethod
    def read(cls, file_path):
        return tf.io.read_file(file_path)

    @classmethod
    def decode(cls, serialized_image):
        image = tf.io.decode_image(serialized_image, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=cls.IMG_DTYPE)
        image.set_shape(cls.RAW_IMG_SHAPE)
        return image

    @classmethod
    def preprocess(cls, image):
        image = tf.image.resize(image, size=cls.IMG_SHAPE[:-1])
        return image

    # --- Methods. ---

    def _build(self):
        """Builds tf.datasets.Dataset for the underlying datasets resources."""
        # Get file paths and determine dataset size.
        filepaths = glob.glob(os.path.join(self.data_dir, self.name, "*.JPEG"))
        if self.shuffle:
            random.shuffle(filepaths)
        if self.max_size is not None:
            filepaths = filepaths[: self.max_size]
        self.size = len(filepaths)

        # Build the tf.data.Dataset.
        self.dataset = tf.data.Dataset.from_tensor_slices(filepaths)

        # Read and cache (if necessary).
        self.dataset = self.dataset.map(self.read, num_parallel_calls=1)
        if self.cache:
            self.dataset = self.dataset.cache()

        # Repeat, batch, prefetch.
        self.dataset = (
            self.dataset.repeat()
            .batch(self.size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        # Build string handles.
        self.iterator = tf.data.make_one_shot_iterator(self.dataset)
        self.string_handle = self.iterator.string_handle()

    def initialize(self, **kwargs):
        raise NotImplementedError(
            "Categories must be initialized jointly by the data source."
        )


class MiniImageNetDataSource(base.DataSource):
    """Data source for mini-ImageNet."""

    NUM_CATEGORIES = 100

    def __init__(
        self,
        data_dir: str,
        cache: bool = True,
        max_size: Optional[int] = None,
        shuffle_data: bool = True,
        name: Optional[str] = None,
    ):
        super(MiniImageNetDataSource, self).__init__(
            data_dir, name=(name or self.__class__.__name__)
        )
        self.cache = cache
        self.max_size = max_size
        self.shuffle_data = shuffle_data

        # Internals.
        self.data = None
        self.string_handles = None

    # --- Properties. ---

    @property
    def raw_data_shapes(self):
        return MiniImageNetCategory.RAW_SHAPE

    @property
    def raw_data_types(self):
        return MiniImageNetCategory.RAW_DTYPE

    @property
    def data_shapes(self):
        return MiniImageNetCategory.IMG_SHAPE

    @property
    def data_types(self):
        return MiniImageNetCategory.IMG_DTYPE

    # --- Methods. ---

    def read_categories(self, dir_path):
        """Reads categories from the provided directory."""
        return [
            MiniImageNetCategory(
                data_dir=dir_path,
                cache=self.cache,
                max_size=self.max_size,
                shuffle=self.shuffle_data,
                name=name,
            )
            for name in os.listdir(dir_path)
            if os.path.isdir(os.path.join(dir_path, name))
        ]

    def __getitem__(self, set_name):
        """Returns string handles for the corresponding set of the data."""
        return self.string_handles[set_name]

    def _build(self):
        """Build train, valid, and test categories."""
        self.data = {}
        for set_name in ["train", "valid", "test"]:
            logger.debug(f"Building {self.name}: {set_name} categories...")
            self.data[set_name] = self.read_categories(
                os.path.join(self.data_dir, set_name)
            )
            for category in self.data[set_name]:
                category.build()

    def initialize(self, sess: Optional[tf.Session] = None, **kwargs):
        """Initialize all categories."""
        if sess is None:
            sess = tf.get_default_session()
        self.string_handles = {}
        for set_name in ["train", "valid", "test"]:
            logger.debug(f"Initializing {self.name}: {set_name} categories...")
            self.string_handles[set_name] = sess.run(
                [category.string_handle for category in self.data[set_name]]
            )


class MiniImageNetDataset(base.ClfDataset):
    """Implements mini-ImageNet-specific preprocessing functionality."""

    def __init__(
        self,
        num_classes: int,
        data_shapes: tf.TensorShape,
        data_types: tf.DType,
        name: Optional[str] = None,
        **_unused_kwargs,
    ):
        super(MiniImageNetDataset, self).__init__(
            num_classes=num_classes,
            data_shapes=data_shapes,
            data_types=data_types,
            name=(name or self.__class__.__name__),
        )

        # Internals.
        self.string_handle_phs = None

    def _build(self):
        """Builds data tensors for each class by instantiating an iterator."""
        data_tensors = []
        string_handle_phs = []
        for k in range(self.num_classes):
            string_handle_ph = tf.placeholder(
                tf.string, shape=[], name=f"iterator_handle_class_{k}"
            )
            iterator = tf.data.Iterator.from_string_handle(
                string_handle_ph,
                output_types=self.data_types,
                output_shapes=self.data_shapes,
            )
            data_tensor = iterator.get_next()
            data_tensors.append(data_tensor)
            string_handle_phs.append(string_handle_ph)
        self.data_tensors = tuple(data_tensors)
        self.string_handle_phs = tuple(string_handle_phs)

    def get_preprocessor(self) -> Optional[Callable]:
        """Returns a preprocessor for the images."""

        def _inner(args):
            serialized_image, _ = args
            image = MiniImageNetCategory.decode(serialized_image)
            image = MiniImageNetCategory.preprocess(image)
            return image

        return _inner

    def get_feed_list(self, string_handles: Tuple[str]) -> List[Tuple[tf.Tensor, str]]:
        """Returns tensors with data and a feed dict with dependencies."""
        assert len(string_handles) == self.num_classes
        return list(zip(self.string_handle_phs, string_handles))


class MiniImageNetMetaDataset(base.ClfMetaDataset):
    """A meta-dataset that samples mini-ImageNet datasets."""

    Dataset = MiniImageNetDataset

    def __init__(
        self,
        data_source: MiniImageNetDataSource,
        batch_size: int,
        num_classes: int,
        set_name: str,
        name: Optional[str] = None,
        seed: Optional[int] = 42,
        **kwargs,
    ):
        super(MiniImageNetMetaDataset, self).__init__(
            data_source=data_source,
            batch_size=batch_size,
            num_classes=num_classes,
            name=(name or self.__class__.__name__),
            seed=seed,
            **kwargs,
        )
        self.set_name = set_name

        # Internals.
        self._rng = np.random.RandomState(seed)

    def get_feed_batch(
        self, requests: Optional[Tuple[Any, ...]] = None, replace: bool = False
    ) -> Tuple[Tuple[Any, ...], List[List[Tuple[tf.Tensor, Any]]]]:
        """Returns a feed list for the requested meta-batch of datasets."""
        # If a batch of requests is not provided, generate from the data source.
        if requests is None:
            requests = tuple(
                tuple(
                    self._rng.choice(
                        len(self.data_source[self.set_name]),
                        size=self.num_classes,
                        replace=replace,
                    )
                )
                for _ in range(self.batch_size)
            )
        elif len(requests) != self.batch_size:
            raise ValueError(
                f"The number of requests ({len(requests)}) does not match "
                f"the meta batch size ({self.batch_size})."
            )
        # Get feed dicts for each request.
        feed_lists = []
        for n, ids in enumerate(requests):
            category_handles = tuple(self.data_source[self.set_name][i] for i in ids)
            feed_lists.append(self.dataset_batch[n].get_feed_list(category_handles))
        return requests, feed_lists
