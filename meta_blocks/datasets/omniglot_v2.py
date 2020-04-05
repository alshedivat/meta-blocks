"""The Omniglot dataset."""

import glob
import logging
import os
import random

from PIL import Image
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import tensorflow.compat.v1 as tf

from meta_blocks.datasets import base

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()

__all__ = [
    "OmniglotCategory",
    "OmniglotDataPool",
    "OmniglotDataset",
    "OmniglotMetaDataset",
]


def get_categories(
    data_dir,
    num_train_categories=1000,
    num_valid_categories=200,
    num_test_categories=463,
    shuffle=True,
):
    logger.info("Reading Omniglot categories...")
    categories = []
    for alphabet_name in sorted(os.listdir(data_dir)):
        alphabet_dir = os.path.join(data_dir, alphabet_name)
        if not os.path.isdir(alphabet_dir):
            continue
        for name in sorted(os.listdir(alphabet_dir)):
            if not os.path.isdir(os.path.join(alphabet_dir, name)):
                continue
            if not name.startswith("character"):
                continue
            char_name = f"{alphabet_name}_{name}"
            char_dir = os.path.join(data_dir, alphabet_name, name)
            categories.append(OmniglotCategory(char_dir, char_name))
    if shuffle:
        random.shuffle(categories)
    categories_dict = {"train": categories[:num_train_categories]}
    if num_valid_categories is not None:
        categories_dict["valid"] = categories[num_train_categories:][
            :num_valid_categories
        ]
    if num_test_categories is not None:
        categories_dict["test"] = categories[num_train_categories:][
            -num_test_categories:
        ]
    return categories_dict


class OmniglotCategory(base.Category):
    """Provides tf.data.Dataset that represents a single Omniglot category."""

    RAW_IMG_SHAPE = (105, 105, 1)
    PREPROC_IMG_SHAPE = (28, 28, 1)

    def __init__(self, data_dir=None, name=None):
        super(OmniglotCategory, self).__init__(data_dir, name)

        # Internals.
        self._dataset = None

    @property
    def dataset(self) -> np.ndarray:
        return self._dataset

    @classmethod
    def preprocess(cls, image, rotation=0):
        image = tf.image.rot90(image, k=rotation)
        return image

    def _build(self, max_size=None, shuffle=True):
        # Infer dataset size.
        filepaths = glob.glob(os.path.join(self.data_dir, "*.png"))
        if shuffle:
            random.shuffle(filepaths)
        if max_size is not None:
            filepaths = filepaths[:max_size]
        self._size = len(filepaths)

        # Load data.
        self._dataset = []
        for fpath in filepaths:
            with open(fpath, "rb") as fp:
                image = Image.open(fp).resize(self.PREPROC_IMG_SHAPE[:-1])
                image = np.array(image).astype(np.float32)
                self._dataset.append(np.expand_dims(image, axis=-1))
        self._dataset = np.stack(self._dataset)

        return self


class OmniglotDataPool(base.DataPool):
    """Manages Omniglot categories."""

    def __init__(
        self, categories: List[OmniglotCategory], name: str = "OmniglotDataPool"
    ):
        super(OmniglotDataPool, self).__init__(categories, name=name)

    @property
    def output_types(self):
        return tf.float32

    @property
    def output_shapes(self):
        return OmniglotCategory.PREPROC_IMG_SHAPE

    def initialize(self, sess):
        """Initializes all managed data categories within the given session."""
        return self


class OmniglotDataset(base.Dataset):
    """Implements Omniglot-specific preprocessing functionality."""

    def __init__(
        self,
        num_classes: int,
        rotations: Tuple[int] = (0,),
        name: str = "OmniglotDataset",
    ):
        self.name = name
        self.num_classes = num_classes
        self._rotations = rotations

        # Internals.
        self._data_tensors = None
        self._rotations_ph = None

        self.built = False

    def build(self, output_types, output_shapes):
        if not self.built:
            data_tensors = []
            with tf.name_scope(self.name):
                # Data placeholders for each class.
                for k in range(self.num_classes):
                    data_ph = tf.placeholder(
                        dtype=output_types,
                        shape=(None,) + output_shapes,
                        name=f"data_class_{k}",
                    )
                    data_tensors.append(data_ph)
                # Rotations.
                self._rotations_ph = tf.placeholder(
                    tf.int32, shape=[None], name="rotations"
                )
            self._data_tensors = tuple(data_tensors)
            self.built = True
        return self

    def get_preprocessor(self) -> Optional[Callable]:
        def _inner(args):
            image, label = args
            image = OmniglotCategory.preprocess(
                image, rotation=self._rotations_ph[label]
            )
            return image

        if len(self._rotations) > 1:
            return _inner

    def get_feed_list(
        self, data_arrays: Tuple[np.ndarray]
    ) -> List[Tuple[tf.Tensor, Any]]:
        feed_list = [
            (ph, array) for array, ph in zip(data_arrays, self._data_tensors)
        ]
        rotations = random.choices(self._rotations, k=self.num_classes)
        feed_list.append((self._rotations_ph, rotations))
        return feed_list


class OmniglotMetaDataset(base.MetaDataset):
    """A meta-dataset that samples Omniglot datasets."""

    Dataset = OmniglotDataset

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
                    [self.data_pool.categories[i].dataset for i in category_ids]
                )
            )
        return feed_lists
