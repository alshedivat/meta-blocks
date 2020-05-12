"""The Mini-ImageNet dataset."""

import glob
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow.compat.v1 as tf

from meta_blocks.datasets import base, utils

logger = logging.getLogger(__name__)

__all__ = [
    "MiniImageNetCategory",
    "MiniImageNetDataSource",
    "MiniImageNetDataset",
    "MiniImageNetMetaDataset",
]

# Types.
DatasetRequest = Tuple[
    # Data source IDs that represent dataset classes.
    np.ndarray,
    # A tuple of selected image ids for each data class.
    Tuple[np.ndarray],
]
FeedList = List[Tuple[tf.Tensor, np.ndarray]]


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
        self, data_dir: str, already_cached: bool = False, name: Optional[str] = None
    ):
        super(MiniImageNetCategory, self).__init__(
            data_dir, name=(name or self.__class__.__name__)
        )
        self.already_cached = already_cached

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
        image = tf.io.decode_image(serialized_image, channels=cls.IMG_SHAPE[-1])
        image = tf.image.convert_image_dtype(image, dtype=cls.IMG_DTYPE)
        image.set_shape(cls.IMG_SHAPE)
        image = image / 0xFF  # Rescale image to [0, 1].
        return image

    # --- Methods. ---

    def _build(self):
        """Builds tf.data.Dataset for the underlying data resources."""
        # Get file paths.
        file_paths = glob.glob(os.path.join(self.data_dir, self.name, "*.jpg"))
        self.size = len(file_paths)

        # Build the tf.data.Dataset.
        self.dataset = (
            tf.data.Dataset.from_tensor_slices(file_paths)
            .map(self.read, num_parallel_calls=8, deterministic=True)
            .map(self.decode, num_parallel_calls=8, deterministic=True)
            .batch(self.size)
            .repeat()
        )


class MiniImageNetDataSource(base.DataSource):
    """Data source for mini-ImageNet."""

    NUM_CATEGORIES = 100

    def __init__(self, data_dir: str, name: Optional[str] = None):
        super(MiniImageNetDataSource, self).__init__(
            data_dir, name=(name or self.__class__.__name__)
        )

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
            MiniImageNetCategory(data_dir=dir_path, name=name)
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
        data_source_size: Optional[int] = None,
        name: Optional[str] = None,
        **_unused_kwargs,
    ):
        super(MiniImageNetDataset, self).__init__(
            num_classes=num_classes, name=(name or self.__class__.__name__)
        )
        self.data_sources = data_sources
        self.data_source_size = data_source_size

        # Internal.
        self.dataset = None
        self.data_tensors = None
        self.data_source_ids = None
        self.selected_ids_phs = None

    # --- Methods. ---

    def build(self):
        """Builds the dataset in the correct namespace."""
        # Ensure dataset construction ops are placed on the CPU.
        with tf.device("CPU:0"):
            return super(MiniImageNetDataset, self).build()

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
        # Build selected ids placeholders.
        self.selected_ids_phs = tuple(
            tf.placeholder(
                dtype=tf.int32,
                shape=(self.data_source_size,),
                name=f"{self.name}_class{k}_ids",
            )
            for k in range(self.num_classes)
        )
        # Build the dataset from the underlying data sources.
        self.dataset = (
            tf.data.experimental.choose_from_datasets(
                datasets=[c.dataset for c in self.data_sources],
                choice_dataset=self._build_category_choice_dataset(),
            )
            .batch(self.num_classes)
            .prefetch(1)
        )
        data = self.dataset.make_one_shot_iterator().get_next()
        # Tuple of <float32> [1, None, **MiniImageNetCategory.IMG_SHAPE].
        data_tensors = tf.split(data, self.num_classes)
        # Tuple of <float32> [None, **MiniImageNetCategory.IMG_SHAPE].
        # Note: do not use tf.squeeze; the results will be of unknown shape.
        data_tensors = tuple(map(lambda x: x[0], data_tensors))
        # Select the specified permutation of elements from data tensors.
        self.data_tensors = tuple(
            tf.gather(x, ids, axis=0)
            for x, ids in zip(data_tensors, self.selected_ids_phs)
        )
        # Determine dataset size.
        data_source_size = self.data_source_size or self.data_sources[0].size
        self._size = self.num_classes * data_source_size

    def set_data_source_ids(self, data_source_ids: Tuple[np.ndarray, ...]):
        self.data_source_ids = data_source_ids

    def get_feed_list(self, selected_ids: Tuple[np.ndarray]) -> FeedList:
        """Returns a feed list of for the internal data placeholders."""
        feed_list = list(zip(self.selected_ids_phs, selected_ids))
        return feed_list


class MiniImageNetMetaDataset(base.ClfMetaDataset):
    """A meta-dataset that samples mini-ImageNet datasets."""

    def __init__(
        self,
        batch_size: int,
        num_classes: int,
        data_sources: List[MiniImageNetCategory],
        data_source_size: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super(MiniImageNetMetaDataset, self).__init__(
            batch_size=batch_size,
            num_classes=num_classes,
            data_sources=data_sources,
            name=(name or self.__class__.__name__),
        )
        self.data_source_size = data_source_size

        # Random state must be set globally.
        self._rng = np.random

        # Internals.
        self.dataset_batch = None

    # --- Methods. ---

    def _build(self):
        """Build datasets in the dataset batch."""
        self.dataset_batch = tuple(
            MiniImageNetDataset(
                num_classes=self.num_classes,
                data_sources=self.data_sources,
                data_source_size=self.data_source_size,
                name=f"Dataset{i}",
            ).build()
            for i in range(self.batch_size)
        )

    def get_feed_list(self, selected_masks: Dict[int, np.ndarray]) -> FeedList:
        """Returns a feed list of for the internal data source placeholders."""
        feed_list = []
        for i, selected_mask in selected_masks.items():
            feed_list.extend(self.data_sources[i].get_feed_list(selected_mask))
        return feed_list

    def request_datasets(
        self,
        requests_batch: Optional[Tuple[DatasetRequest, ...]] = None,
        unique_classes: bool = True,
    ) -> Tuple[Tuple[DatasetRequest, ...], FeedList]:
        """Returns a feed list for the requested meta-batch of datasets."""
        # Generate a batch of requests is not provided.
        if requests_batch is None:
            requests_batch = tuple(
                utils.generate_dataset_request(
                    data_sources=self.data_sources,
                    num_classes=self.num_classes,
                    unique_classes=unique_classes,
                    data_source_size=self.data_source_size,
                )
                for _ in range(self.batch_size)
            )
        elif len(requests_batch) != self.batch_size:
            raise ValueError(
                f"The number of requests ({len(requests_batch)}) does not match "
                f"the meta batch size ({self.batch_size})."
            )
        # Use requests to set category ids for each dataset in the batch.
        feed_list = []
        for dataset, request in zip(self.dataset_batch, requests_batch):
            data_source_ids, selected_ids = request
            dataset.set_data_source_ids(data_source_ids=data_source_ids)
            feed_list.extend(dataset.get_feed_list(selected_ids=selected_ids))
        return requests_batch, feed_list
