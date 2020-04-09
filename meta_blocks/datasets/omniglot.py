"""
Loading and augmenting the Omniglot dataset.

To use these APIs, you should prepare a directory that
contains all of the alphabets from both images_background
and images_evaluation.
"""

import glob
import logging
import os
import random

import tensorflow.compat.v1 as tf

from meta_blocks.datasets import base, utils

logger = logging.getLogger(__name__)

__all__ = ["OmniglotCategory", "OmniglotDataset", "OmniglotMetaDataset"]


def read_dataset(
    data_dir,
    num_train_categories=1000,
    num_valid_categories=200,
    num_test_categories=463,
):
    """Iterate over the characters in a data directory.
    The dataset is unaugmented and not split up into training and test sets.

    Parameters
    ----------
    data_dir : str
        A directory of alphabet directories.

    num_train_categories : int, optional (default=1000)

    num_valid_categories : int, optional (default=200)

    num_test_categories : int, optional (default=463)

    Returns
    -------
    train_test_val_set : dict
        An iterable over Characters.
    """

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
            c = OmniglotCategory(data_dir, os.path.join(alphabet_name, name))
            categories.append(c)
    random.shuffle(categories)
    return {
        "train": categories[:num_train_categories],
        "valid": categories[num_train_categories:][:num_valid_categories],
        "test": categories[num_train_categories:][-num_test_categories:],
    }


class OmniglotCategory(base.Category):
    """A single character class.

    Parameters
    ----------
    data_dir : str
        The data directory.

    name : str
        Name of the category.

    maybe_gen_tfrecords : bool, optional (default=False)
        Whether to generate TF records. Deprecated.
    """

    def __init__(self, data_dir, name, maybe_gen_tfrecords=False):

        super(OmniglotCategory, self).__init__(data_dir, name)
        self._maybe_gen_tfrecords = maybe_gen_tfrecords

    def maybe_generate_tfrecords(self, tfrecord_path):
        """Generates TFRecords for the corresponding data."""
        # Get file paths and determine dataset size.
        filepaths = glob.glob(os.path.join(self.data_dir, self.name, "*.png"))
        self._dataset_size = len(filepaths)

        # Generate TFRecords, if necessary.
        if os.path.exists(tfrecord_path):
            logger.debug(
                f"Category {self.name} in {self.data_dir} "
                f"is already preprocessed. Skipping..."
            )
            return
        logger.debug(
            f"Generating TFRecords for category {self.name} " f"in {self.data_dir}..."
        )

        # Read images and construct features.
        features = []
        for fpath in sorted(filepaths):
            with open(fpath, "rb") as fp:
                feature = {"image_raw": utils.bytes_feature_list([fp.read()])}
                features.append(feature)

        # Write features to a TFRecords file.
        with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
            for feature in features:
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

        logger.info("Done.")

    def build(
        self,
        size=None,
        shuffle=False,
        num_parallel_reads=16,
        shuffle_buffer_size=100,
        seed=None,
        **kwargs,
    ):
        """Builds tf.data.Dataset for the underlying data resources."""
        tfrecord_path = os.path.join(self.data_dir, f"{self.name}.tfrecords")

        # Generate TFRecords.
        if self._maybe_gen_tfrecords:
            self.maybe_generate_tfrecords(tfrecord_path)
        else:
            self._dataset_size = 20

        # Build the tf.data.Dataset.
        self._dataset = tf.data.TFRecordDataset(
            tfrecord_path, num_parallel_reads=num_parallel_reads
        )
        if shuffle:
            self._dataset = self._dataset.shuffle(shuffle_buffer_size, seed=seed)
        self._size = (
            self._dataset_size if size is None else min(size, self._dataset_size)
        )
        self._dataset = self._dataset.repeat().batch(self._size)

        # Build string handles.
        iterator = tf.data.make_one_shot_iterator(self._dataset)
        self._string_handle_tensor = iterator.string_handle()

        return self


class OmniglotDataset(base.Dataset):
    """A dataset that implements Omniglot-speicfic data preprocessing."""

    RAW_IMG_SHAPE = (105, 105, 1)
    PREPROC_IMG_SHAPE = (28, 28, 1)

    def __init__(self, categories, rotations=(0,), **kwargs):
        self.rotations = tuple(rotations)
        super(OmniglotDataset, self).__init__(categories, **kwargs)

    def _get_preprocess_kwargs(self):
        return tuple(
            {"rotation": random.sample(self.rotations, 1)[0]} for _ in self._categories
        )

    @classmethod
    def build_preprocess_kwarg_phs(cls, num_classes):
        """Builds kwarg placeholders used for preprocessing."""
        return tuple(
            {
                "rotation": tf.placeholder(
                    dtype=tf.int32, shape=[], name=f"class_{i}_rotation"
                )
            }
            for i in range(num_classes)
        )

    @classmethod
    def preprocess(cls, example, rotation=0, **kwargs):
        image = utils.deserialize_image(example, channels=1, shape=cls.RAW_IMG_SHAPE)
        image = tf.image.resize(image, size=cls.PREPROC_IMG_SHAPE[:-1])
        image = tf.image.rot90(image, k=rotation)
        return image


class OmniglotMetaDataset(base.MetaDataset):
    """A meta-dataset with omniglot-specific data preprocessing."""

    Dataset = OmniglotDataset

    def __init__(
        self, categories, num_classes, max_distinct_datasets=None, rotations=(0,)
    ):
        super(OmniglotMetaDataset, self).__init__(
            categories=categories,
            num_classes=num_classes,
            max_distinct_datasets=max_distinct_datasets,
        )
        self._dataset_kwargs = {"rotations": rotations}
