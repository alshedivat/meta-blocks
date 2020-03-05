"""Models for Kkanji datasets."""

import tensorflow.compat.v1 as tf

from . import base

__all__ = ["StandardFeedForwardModel", "EmbedFeedForwardModel", "ProtoModel"]

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


class _KkanjiModelMixin(base.Model):
    """A mixin base class that defines Kkanji-specific placeholders."""

    @staticmethod
    def get_data_placeholders():
        inputs_ph = tf.placeholder(
            dtype=tf.float32, shape=(None, 64, 64), name="inputs"
        )
        labels_ph = tf.placeholder(dtype=tf.int32, shape=(None,), name="labels")
        return inputs_ph, labels_ph


class StandardFeedForwardModel(base.FeedForwardModel, _KkanjiModelMixin):
    """The standard feed-forward model for Kkanji dataset."""

    def __init__(self, num_classes, name="StandardFeedForwardModel", **kwargs):
        base.FeedForwardModel.__init__(self, num_classes, name=name, **kwargs)

    def _build_embeddings(self, input_ph):
        return input_ph

    def _build_logits(self, embeddings):
        out = tf.reshape(embeddings, (-1, 64, 64, 1))
        for _ in range(4):
            out = tf.layers.conv2d(out, 64, 3, strides=2, padding="same")
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
        out = tf.layers.flatten(out)
        return tf.layers.dense(out, self.num_classes)


class EmbedFeedForwardModel(base.FeedForwardModel, _KkanjiModelMixin):
    """The standard feed-forward model for Kkanji dataset."""

    EMB_SIZE = 256

    def __init__(self, num_classes, name="EmbedFeedForwardModel", **kwargs):
        base.FeedForwardModel.__init__(self, num_classes, name=name, **kwargs)

    def _build_embeddings(self, input_ph):
        out = tf.reshape(input_ph, (-1, 64, 64, 1))
        for _ in range(3):
            out = tf.layers.conv2d(out, 64, 3, strides=2, padding="same")
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
        out = tf.layers.flatten(out)
        return tf.layers.dense(out, self.EMB_SIZE)

    def _build_logits(self, embeddings):
        return tf.layers.dense(embeddings, self.num_classes)


class ProtoModel(base.ProtoModel, _KkanjiModelMixin):
    """The standard feed-forward model for Kkanji dataset."""

    EMB_SIZE = 256

    def _build_embeddings(self, input_ph):
        out = tf.reshape(input_ph, (-1, 64, 64, 1))
        for _ in range(3):
            out = tf.layers.conv2d(out, 64, 3, strides=2, padding="same")
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
        out = tf.layers.flatten(out)
        return tf.layers.dense(out, self.EMB_SIZE)


def get(name, build=True, **kwargs):
    if name == "standard_ff":
        model = StandardFeedForwardModel(**kwargs)
    elif name == "embed_ff":
        model = EmbedFeedForwardModel(**kwargs)
    elif name == "proto":
        model = ProtoModel(**kwargs)
    else:
        raise ValueError(f"Unsupported model: {name}")
    if build:
        model.build()
    return model
