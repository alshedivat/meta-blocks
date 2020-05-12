"""The Omniglot dataset."""

import glob
import logging
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image, PngImagePlugin

from meta_blocks.datasets import base, utils

logger = logging.getLogger(__name__)

# Disable DEBUG output from PIL.PngImagePlugin.
pil_logger = logging.getLogger(PngImagePlugin.__name__)
pil_logger.setLevel(logging.INFO)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()

__all__ = [
    "OmniglotCharacter",
    "OmniglotDataSource",
    "OmniglotDataset",
    "OmniglotMetaDataset",
]

# Types.
DatasetRequest = Tuple[
    # Data source IDs that represent dataset classes.
    np.ndarray,
    # A tuple of selected image ids for each data class.
    Tuple[np.ndarray],
]
FeedList = List[Tuple[tf.Tensor, np.ndarray]]


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
        self.size = None

    # --- Properties. ---

    @property
    def data_shapes(self):
        return self.IMG_SHAPE

    @property
    def data_types(self):
        return self.IMG_DTYPE

    # --- Methods. ---

    def _build(self):
        # Infer dataset size.
        file_paths = glob.glob(os.path.join(self.data_dir, "*.png"))
        self.size = len(file_paths)
        # Load data.
        data = []
        for fpath in file_paths:
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
        rotations: Optional[Tuple[int]] = None,
        shuffle: bool = True,
        name: Optional[str] = None,
    ):
        super(OmniglotDataSource, self).__init__(
            data_dir=data_dir, name=(name or self.__class__.__name__)
        )
        self.num_train = num_train_categories
        self.num_valid = num_valid_categories
        self.num_test = num_test_categories
        self.rotations = rotations
        self.shuffle = shuffle

        # Internals.
        self.data = None

    @property
    def data_shapes(self):
        return OmniglotCharacter.IMG_SHAPE

    @property
    def data_types(self):
        return OmniglotCharacter.IMG_DTYPE

    # --- Methods. ---

    def __getitem__(self, set_name):
        """Returns the corresponding set of the data."""
        return self.data[set_name]

    def _build(self):
        """Loads train, valid, and test categories."""
        logger.debug(f"Building {self.name}...")
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
                char_dir = os.path.join(self.data_dir, alphabet_name, name)
                char_name = f"{alphabet_name}_{name}".replace("(", "").replace(")", "")
                characters.append(
                    OmniglotCharacter(data_dir=char_dir, name=char_name).build()
                )
        if self.shuffle:
            random.shuffle(characters)
        self.data = {
            "train": tuple(characters[: self.num_train]),
            "valid": tuple(characters[self.num_train :][: self.num_valid]),
            "test": tuple(characters[self.num_test :][-self.num_test :]),
        }
        # Expand training characters with their rotated versions.
        if self.rotations is not None:
            rotated_train_characters = []
            for rot in self.rotations:
                for char in characters[: self.num_train]:
                    rotated_train_characters.append(
                        OmniglotCharacter(
                            data_dir=char.data_dir,
                            rotation=rot,
                            name=f"{char.name}_{rot}",
                        ).build()
                    )
            self.data["train"] += tuple(rotated_train_characters)


class OmniglotDataset(base.ClfDataset):
    """Implements Omniglot-specific preprocessing functionality."""

    def __init__(
        self,
        num_classes: int,
        data_sources: List[OmniglotCharacter],
        data_source_size: Optional[int] = None,
        name: Optional[str] = None,
        **_unused_kwargs,
    ):
        super(OmniglotDataset, self).__init__(
            num_classes=num_classes, name=(name or self.__class__.__name__)
        )
        self.data_sources = data_sources
        self.data_source_size = data_source_size
        self.data_shapes = self.data_sources[0].data_shapes
        self.data_types = self.data_sources[0].data_types

    def _build(self):
        """Builds data placeholdes for each class."""
        # Build data tensors.
        data_tensors = []
        for k in range(self.num_classes):
            data_ph = tf.placeholder(
                shape=(None,) + self.data_shapes,
                dtype=self.data_types,
                name=f"data_class_{k}",
            )
            data_tensors.append(data_ph)
        self.data_tensors = tuple(data_tensors)
        # Determine dataset size.
        data_source_size = self.data_source_size or self.data_sources[0].size
        self._size = self.num_classes * data_source_size

    def get_feed_list(self, data_arrays: Tuple[np.ndarray, ...]) -> FeedList:
        """Returns a feed list of for the internal data placeholders."""
        assert len(data_arrays) == len(self.data_tensors)
        return list(zip(self.data_tensors, data_arrays))


class OmniglotMetaDataset(base.ClfMetaDataset):
    """A meta-dataset that samples Omniglot datasets."""

    def __init__(
        self,
        batch_size: int,
        num_classes: int,
        data_sources: List[OmniglotCharacter],
        data_source_size: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super(OmniglotMetaDataset, self).__init__(
            batch_size=batch_size,
            num_classes=num_classes,
            data_sources=data_sources,
            name=(name or self.__class__.__name__),
        )
        self.data_source_size = data_source_size

        # Random state must be set globally.
        self._rng = np.random

    def _build(self):
        """Build datasets in the dataset batch."""
        self.dataset_batch = tuple(
            OmniglotDataset(
                num_classes=self.num_classes,
                data_sources=self.data_sources,
                data_source_size=self.data_source_size,
                name=f"Dataset{i}",
            ).build()
            for i in range(self.batch_size)
        )

    def request_datasets(
        self,
        requests_batch: Optional[Tuple[DatasetRequest, ...]] = None,
        unique_classes: bool = True,
    ) -> Tuple[Tuple[DatasetRequest, ...], FeedList]:
        """Returns a feed list for the requested meta-batch of datasets."""
        # If a batch of requests is not provided, generate from the data source.
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
        # Get feed dicts for each request.
        feed_list = []
        for n, (data_source_ids, selected_ids) in enumerate(requests_batch):
            data_arrays = tuple(
                # From each data array take only selected items.
                self.data_sources[i].data[ids]
                for i, ids in zip(data_source_ids, selected_ids)
            )
            feed_list.extend(self.dataset_batch[n].get_feed_list(data_arrays))
        return requests_batch, feed_list
