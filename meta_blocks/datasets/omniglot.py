"""The Omniglot dataset."""

import glob
import logging
import os
import random
from typing import Any, List, Optional, Tuple

import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image

from meta_blocks.datasets import base

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()

__all__ = [
    "OmniglotCharacter",
    "OmniglotDataSource",
    "OmniglotDataset",
    "OmniglotMetaDataset",
]


class OmniglotCharacter(base.DataSource):
    """Represents data source for a single Omniglot character.

    Parameters
    ----------
    data_dir : str
        Path to the directory that contains the character data.

    rotation : int (default: 0)
        Rotation of the character in degrees.

    name : str, optional
        The name of the dataset.
    """

    RAW_IMG_SHAPE = (105, 105, 1)
    IMG_SHAPE = (28, 28, 1)
    IMG_DTYPE = tf.float32

    def __init__(self, data_dir: str, rotation: int = 0, name: Optional[str] = None):
        super(OmniglotCharacter, self).__init__(
            data_dir, name=(name or self.__class__.__name__)
        )
        self.rotation = rotation

        # Internals.
        self.data = None

    # --- Properties. ---

    @property
    def raw_data_shapes(self):
        return self.IMG_SHAPE

    @property
    def raw_data_types(self):
        return self.IMG_DTYPE

    # --- Methods. ---

    def initialize(self, max_size=None, shuffle=True):
        # Infer dataset size.
        filepaths = glob.glob(os.path.join(self.data_dir, "*.png"))
        if shuffle:
            random.shuffle(filepaths)
        if max_size is not None:
            filepaths = filepaths[:max_size]
        self._size = len(filepaths)
        # Load data.
        data = []
        for fpath in filepaths:
            with open(fpath, "rb") as fp:
                image = Image.open(fp).resize(self.IMG_SHAPE[:-1])
                if self.rotation:
                    image = image.rotate(self.rotation)
                image = np.array(image).astype(np.float32)
                data.append(np.expand_dims(image, axis=-1))
        self.data = np.stack(data)


class OmniglotDataSource(base.DataSource):
    """Data source for Omniglot data."""

    NUM_CATEGORIES = 1663

    def __init__(
        self,
        data_dir: str,
        num_train_categories: int = 1000,
        num_valid_categories: int = 200,
        num_test_categories: int = 463,
        max_category_size: Optional[int] = None,
        rotations: Optional[Tuple[int]] = None,
        shuffle_categories: bool = True,
        shuffle_data: bool = True,
        name: Optional[str] = None,
    ):
        super(OmniglotDataSource, self).__init__(
            data_dir=data_dir, name=(name or self.__class__.__name__)
        )
        self.num_train_categories = num_train_categories
        self.num_valid_categories = num_valid_categories
        self.num_test_categories = num_test_categories
        self.max_category_size = max_category_size
        self.rotations = rotations
        self.shuffle_categories = shuffle_categories
        self.shuffle_data = shuffle_data

        # Internals.
        self.data = None

    @property
    def raw_data_shapes(self):
        return OmniglotCharacter.IMG_SHAPE

    @property
    def raw_data_types(self):
        return OmniglotCharacter.IMG_DTYPE

    # --- Methods. ---

    def __getitem__(self, set_name):
        """Returns the corresponding set of the data."""
        return self.data[set_name]

    def initialize(self):
        """Loads train, valid, and test categories."""
        logger.debug(f"Initializing {self.name}...")
        characters = []
        for alphabet_name in sorted(os.listdir(self.data_dir)):
            alphabet_dir = os.path.join(self.data_dir, alphabet_name)
            if not os.path.isdir(alphabet_dir):
                continue
            for name in sorted(os.listdir(alphabet_dir)):
                if not os.path.isdir(os.path.join(alphabet_dir, name)):
                    continue
                if not name.startswith("character"):
                    continue
                char_name = f"{alphabet_name}_{name}"
                char_dir = os.path.join(self.data_dir, alphabet_name, name)
                char = OmniglotCharacter(char_dir, name=char_name)
                char.initialize(
                    max_size=self.max_category_size, shuffle=self.shuffle_data
                )
                characters.append(char)
        if self.shuffle_categories:
            random.shuffle(characters)
        self.data = {
            "train": tuple(characters[: self.num_train_categories]),
            "valid": tuple(
                characters[self.num_train_categories :][: self.num_valid_categories]
            ),
            "test": tuple(
                characters[self.num_test_categories :][-self.num_test_categories :]
            ),
        }
        # Expand training characters with their rotated versions.
        if self.rotations is not None:
            rotated_train_characters = []
            for rot in self.rotations:
                for char in self.data["train"]:
                    rot_char = OmniglotCharacter(
                        data_dir=char.data_dir, rotation=rot, name=f"{char.name}_{rot}"
                    )
                    rot_char.initialize(
                        max_size=self.max_category_size, shuffle=self.shuffle_data
                    )
                    rotated_train_characters.append(rot_char)
            self.data["train"] = self.data["train"] + tuple(rotated_train_characters)


class OmniglotDataset(base.ClfDataset):
    """Implements Omniglot-specific preprocessing functionality."""

    def __init__(
        self,
        num_classes: int,
        data_shapes: tf.TensorShape,
        data_types: tf.DType,
        name: Optional[str] = None,
        **_unused_kwargs,
    ):
        super(OmniglotDataset, self).__init__(
            num_classes=num_classes,
            data_shapes=data_shapes,
            data_types=data_types,
            name=(name or self.__class__.__name__),
        )

    def _build(self):
        """Builds data placeholdes for each class."""
        data_tensors = []
        for k in range(self.num_classes):
            data_ph = tf.placeholder(
                shape=(None,) + self.data_shapes,
                dtype=self.data_types,
                name=f"data_class_{k}",
            )
            data_tensors.append(data_ph)
        self.data_tensors = tuple(data_tensors)

    def get_feed_list(
        self, data_arrays: Tuple[np.ndarray, ...]
    ) -> List[Tuple[tf.Tensor, np.ndarray]]:
        assert len(data_arrays) == len(self.data_tensors)
        return list(zip(self.data_tensors, data_arrays))


class OmniglotMetaDataset(base.ClfMetaDataset):
    """A meta-dataset that samples Omniglot datasets."""

    Dataset = OmniglotDataset

    def __init__(
        self,
        data_source: OmniglotDataSource,
        batch_size: int,
        num_classes: int,
        set_name: str,
        name: Optional[str] = None,
        seed: Optional[int] = 42,
        **kwargs,
    ):
        super(OmniglotMetaDataset, self).__init__(
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
            data_arrays = tuple(self.data_source[self.set_name][i].data for i in ids)
            feed_lists.append(self.dataset_batch[n].get_feed_list(data_arrays))
        return requests, feed_lists
