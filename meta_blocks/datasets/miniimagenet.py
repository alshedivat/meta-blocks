"""The Mini-ImageNet dataset."""

import glob
import logging
import os
import random
from concurrent import futures
from typing import List, Optional, Tuple

import filelock
import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image

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

    name : str, optional
        The name of the dataset.
    """

    IMG_SHAPE = (84, 84, 3)
    IMG_DTYPE = tf.float32

    def __init__(
        self,
        data_dir: str,
        shuffle: bool = True,
        max_size: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super(MiniImageNetCategory, self).__init__(
            data_dir, name=(name or self.__class__.__name__)
        )
        self.shuffle = shuffle
        self.max_size = max_size

        # Internals.
        self.dataset = None
        self.size = None

    # --- Properties. ---

    @property
    def data_shapes(self):
        return self.IMG_SHAPE

    @property
    def data_types(self):
        return self.IMG_DTYPE

    # --- Class methods. ---

    @classmethod
    def read(cls, file_path):
        return tf.io.read_file(file_path)

    @classmethod
    def decode(cls, serialized_image):
        image = tf.io.decode_image(serialized_image, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=cls.IMG_DTYPE)
        image.set_shape(cls.IMG_SHAPE)
        image = image / 0xFF  # Rescale image to [0, 1].
        return image

    # --- Methods. ---

    def _preprocess_and_maybe_cache(self):
        """Creates cached preprocessed images if the necessary."""
        logger.debug(f"Processing and caching {self.name}...")
        file_paths = glob.glob(os.path.join(self.data_dir, self.name, "*.JPEG"))
        filelock_logger = logging.getLogger(filelock.__name__)
        filelock_logger.setLevel(logging.ERROR)

        def _process_filepath(file_path):
            if "resized" not in file_path:
                dir_path = os.path.dirname(file_path)
                file_name = os.path.basename(file_path)
                resized_name = file_name.split(".")[0] + ".resized.JPEG"
                resized_path = os.path.join(dir_path, resized_name)
                with filelock.FileLock(f"{file_path}.lock"):
                    if not os.path.exists(resized_path):
                        with filelock.FileLock(f"{resized_path}.lock"):
                            img = Image.open(file_path).resize(self.IMG_SHAPE[:-1])
                            img.save(resized_path, "JPEG", quality=99)
            else:
                resized_path = file_path
            return resized_path

        # Use multi-threading to speed up IO-bound processing.
        with futures.ThreadPoolExecutor(max_workers=16) as executor:
            resized_paths = list(
                executor.map(_process_filepath, file_paths, chunksize=4)
            )

        return resized_paths

    def _build(self):
        """Builds tf.data.Dataset for the underlying data resources."""
        # Get file paths.
        file_paths = self._preprocess_and_maybe_cache()

        # Shuffle and compute size.
        if self.shuffle:
            random.shuffle(file_paths)
        if self.max_size is not None:
            file_paths = file_paths[: self.max_size]
        self.size = len(file_paths)

        # Build the tf.data.Dataset.
        self.dataset = (
            tf.data.Dataset.from_tensor_slices(file_paths)
            .map(self.read, num_parallel_calls=1)
            .map(self.decode, num_parallel_calls=16)
            .batch(self.size)
            .repeat()
        )


class MiniImageNetDataSource(base.DataSource):
    """Data source for mini-ImageNet."""

    NUM_CATEGORIES = 100

    def __init__(
        self,
        data_dir: str,
        max_size: Optional[int] = None,
        shuffle_data: bool = True,
        name: Optional[str] = None,
    ):
        super(MiniImageNetDataSource, self).__init__(
            data_dir, name=(name or self.__class__.__name__)
        )
        self.max_size = max_size
        self.shuffle_data = shuffle_data

        # Internals.
        self.data = None

    # --- Properties. ---

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
                max_size=self.max_size,
                shuffle=self.shuffle_data,
                name=name,
            )
            for name in os.listdir(dir_path)
            if os.path.isdir(os.path.join(dir_path, name))
        ]

    def __getitem__(self, set_name):
        """Returns string handles for the corresponding set of the data."""
        return self.data[set_name]

    def _build(self):
        """Build train, valid, and test data categories."""
        self.data = {}
        for set_name in ["train", "valid", "test"]:
            logger.debug(f"Building {self.name}: {set_name} categories...")
            self.data[set_name] = self.read_categories(
                os.path.join(self.data_dir, set_name)
            )
            for category in self.data[set_name]:
                category.build()


class MiniImageNetDataset(base.ClfDataset):
    """Implements mini-ImageNet-specific preprocessing functionality."""

    def __init__(
        self,
        num_classes: int,
        data_sources: List[MiniImageNetCategory],
        name: Optional[str] = None,
        **_unused_kwargs,
    ):
        super(MiniImageNetDataset, self).__init__(
            num_classes=num_classes, name=(name or self.__class__.__name__)
        )
        self.data_sources = data_sources

        # Internal.
        self.dataset = None
        self.data_tensors = None
        self.data_source_ids = None

    # --- Methods. ---

    def _build_category_choice_dataset(self):
        """Creates a dataset of category ids used for choosing categories.
        Consistently generates datasets with the classes that correspond to
        `self.category_ids` which can be set externally.
        """

        def _gen_category_choices():
            while True:
                assert isinstance(self.data_source_ids, np.ndarray)
                assert len(self.data_source_ids.shape) == 1
                yield self.data_source_ids

        category_choice_dataset = tf.data.Dataset.from_generator(
            _gen_category_choices,
            output_shapes=tf.TensorShape([self.num_classes]),
            output_types=tf.int64,
        ).unbatch()

        return category_choice_dataset

    def _build(self):
        """Builds data tensors that represent classes."""
        # Build the dataset from the underlying data sources.
        self.dataset = (
            tf.data.experimental.choose_from_datasets(
                datasets=[c.dataset for c in self.data_sources],
                choice_dataset=self._build_category_choice_dataset(),
            )
            .batch(self.num_classes)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        data = self.dataset.make_one_shot_iterator().get_next()
        # Tuple of <float32> [None, **MiniImageNetCategory.IMG_SHAPE].
        self.data_tensors = tuple(map(tf.squeeze, tf.split(data, self.num_classes)))
        # Determine dataset size.
        self._size = self.num_classes * self.data_sources[0].size

    def set_data_source_ids(self, data_source_ids: Tuple[np.ndarray, ...]):
        self.data_source_ids = data_source_ids


class MiniImageNetMetaDataset(base.ClfMetaDataset):
    """A meta-dataset that samples mini-ImageNet datasets."""

    Dataset = MiniImageNetDataset

    def __init__(
        self,
        batch_size: int,
        num_classes: int,
        data_sources: List[MiniImageNetCategory],
        name: Optional[str] = None,
    ):
        super(MiniImageNetMetaDataset, self).__init__(
            batch_size=batch_size,
            num_classes=num_classes,
            data_sources=data_sources,
            name=(name or self.__class__.__name__),
        )

        # Internals.
        self.dataset_batch = None

    # --- Methods. ---

    def _build(self):
        """Build datasets in the dataset batch."""
        self.dataset_batch = tuple(
            self.Dataset(
                num_classes=self.num_classes,
                data_sources=self.data_sources,
                name=f"Dataset{i}",
            ).build()
            for i in range(self.batch_size)
        )

    def request_datasets(
        self,
        requests_batch: Optional[Tuple[np.ndarray, ...]] = None,
        unique_classes: bool = True,
    ) -> Tuple[np.ndarray, ...]:
        """Returns a feed list for the requested meta-batch of datasets."""
        # If a batch of requests is not provided, generate from the data source.
        if requests_batch is None:
            requests_batch = tuple(
                np.random.choice(
                    len(self.data_sources),
                    size=self.num_classes,
                    replace=(not unique_classes),
                )
                for _ in range(self.batch_size)
            )
        elif len(requests_batch) != self.batch_size:
            raise ValueError(
                f"The number of requests ({len(requests_batch)}) does not match "
                f"the meta batch size ({self.batch_size})."
            )
        # Use requests to set category ids for each dataset in the batch.
        for dataset, request in zip(self.dataset_batch, requests_batch):
            dataset.set_data_source_ids(request)
        return requests_batch
