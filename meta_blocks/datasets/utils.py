"""Data utility functions."""

import tensorflow as tf

__all__ = ["bytes_feature_list", "int64_feature_list", "deserialize_image"]


def int64_feature_list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature_list(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def deserialize_image(serialized_example, channels, shape=None):
    """Deserializes a single example that contains an image."""
    schema = {"image_raw": tf.io.FixedLenFeature([], tf.string)}
    features = tf.io.parse_single_example(serialized_example, features=schema)
    image = tf.io.decode_image(features["image_raw"], channels=channels)
    if shape is not None:
        image.set_shape(shape)
    return image
