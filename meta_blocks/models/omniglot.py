"""Models for Omniglot datasets."""

import tensorflow.compat.v1 as tf

from meta_blocks.models import base

__all__ = ["StandardFeedForwardModel", "EmbedFeedForwardModel", "ProtoModel"]

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


class _OmniglotModelMixin(base.Model):
    """A mixin base class that defines Omniglot-specific placeholders."""

    @staticmethod
    def get_data_placeholders():
        inputs_ph = tf.placeholder(
            dtype=tf.float32, shape=(None, 28, 28), name="inputs"
        )
        labels_ph = tf.placeholder(dtype=tf.int32, shape=(None,), name="labels")
        return inputs_ph, labels_ph


class StandardFeedForwardModel(base.FeedForwardModel, _OmniglotModelMixin):
    """The standard feed-forward model for Omniglot dataset."""

    def __init__(self, num_classes, name="StandardFeedForwardModel", **kwargs):
        base.FeedForwardModel.__init__(self, num_classes, name=name, **kwargs)

    def _build_embeddings(self, input_ph):
        return tf.layers.flatten(input_ph)

    def _build_logits(self, embeddings):
        out = tf.reshape(embeddings, (-1, 28, 28, 1))
        for _ in range(4):
            out = tf.layers.conv2d(out, 64, 3, strides=2, padding="same")
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
        out = tf.layers.flatten(out)
        return tf.layers.dense(out, self.num_classes)


class EmbedFeedForwardModel(base.FeedForwardModel, _OmniglotModelMixin):
    """
    The standard feed-forward model for Omniglot dataset with only one final
    dense layer being adaptable.
    """

    EMB_SIZE = 4

    def __init__(self, num_classes, name="EmbedFeedForwardModel", **kwargs):
        base.FeedForwardModel.__init__(self, num_classes, name=name, **kwargs)

    def _build_embeddings(self, input_ph):
        out = tf.reshape(input_ph, (-1, 28, 28, 1))
        for _ in range(4):
            out = tf.layers.conv2d(out, 64, 3, strides=2, padding="same")
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
        return tf.layers.flatten(out)

    def _build_logits(self, embeddings):
        return tf.layers.dense(embeddings, self.num_classes)


class EmbedFeedForwardV2Model(base.FeedForwardModel, _OmniglotModelMixin):
    """
    The standard feed-forward model for Omniglot dataset with a few final layers
    being adaptable.
    """

    EMB_SIZE = 196

    def __init__(self, num_classes, name="EmbedFeedForwardModel", **kwargs):
        base.FeedForwardModel.__init__(self, num_classes, name=name, **kwargs)

    def _build_embeddings(self, input_ph):
        out = tf.reshape(input_ph, (-1, 28, 28, 1))
        for _ in range(2):
            out = tf.layers.conv2d(out, 64, 3, strides=2, padding="same")
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
        out = tf.layers.flatten(out)
        return tf.layers.dense(out, self.EMB_SIZE)

    def _build_logits(self, embeddings):
        out = tf.reshape(embeddings, (-1, 14, 14, 1))
        for _ in range(2):
            out = tf.layers.conv2d(out, 64, 3, strides=2, padding="same")
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
        out = tf.layers.flatten(out)
        return tf.layers.dense(out, self.num_classes)


class ProtoModel(base.ProtoModel, _OmniglotModelMixin):
    """The model that uses prototypes to compute logits for Omniglot dataset."""

    EMB_SIZE = 64

    def _build_embeddings(self, input_ph):
        out = tf.reshape(input_ph, (-1, 28, 28, 1))
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
    elif name == "embed_ff_v2":
        model = EmbedFeedForwardV2Model(**kwargs)
    elif name == "proto":
        model = ProtoModel(**kwargs)
    else:
        raise ValueError(f"Unsupported model: {name}")
    if build:
        model.build()
    return model
