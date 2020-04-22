"""Base models for meta-learning."""

import abc
import logging

import tensorflow.compat.v1 as tf

__all__ = ["ClassificationModel"]

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


class ClassificationModel(abc.ABC):
    """Abstract base class for classification models.

    Subclasses must implement _build and _build_logits methods that build
    corresponding parts of the graph under the model's name scope.

    Parameters
    ----------
    input_shapes : TensorShape
        The shape of the input tensors expected by the model.

    input_types : TensorType
        The type of the input tensors expected by the model.

    name : str
        The name of the model. Used to define the corresponding name scope.
    """

    def __init__(self, input_shapes, input_types, name):
        self.input_shapes = input_shapes
        self.input_types = input_types
        self.name = name

        # Internals.
        self.initial_parameters = None
        self.logits = None
        self.loss = None
        self.preds = None

    @property
    @abc.abstractmethod
    def parameters(self):
        """Returns a list of variables that parameterize the model.
        Must be implemented in a subclass.
        """
        raise NotImplementedError("Abstract Method")

    @property
    @abc.abstractmethod
    def trainable_parameters(self):
        """Returns a list with the subset of variables that are trainable.
        Must be implemented in a subclass.
        """
        raise NotImplementedError("Abstract Method")

    def build(self):
        """Builds the model graph in the correct name scope."""
        with tf.name_scope(self.name):
            self._build()
        return self

    @abc.abstractmethod
    def _build(self):
        """Builds the model graph for prediction.
        Must be implemented in a subclass.
        """
        raise NotImplementedError("Abstract Method")

    def build_logits(self, inputs_ph):
        """Builds the logits for the provided inputs."""
        with tf.name_scope(self.name):
            # <float32> [None, num_classes].
            logits = self._build_logits(inputs_ph)
        return logits

    @abc.abstractmethod
    def _build_logits(self, inputs_ph):
        """Builds a part of the model graph for computing output logits.
        Must be implemented in a subclass.
        """
        raise NotImplementedError("Abstract Method")

    def build_loss(self, inputs_ph, labels_ph):
        """Builds the model loss on top provided data placeholders."""
        with tf.name_scope(self.name):
            # Build logits.
            # <float32> [None, num_classes].
            logits = self.build_logits(inputs_ph)
            # Build loss.
            # <float32> [None].
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels_ph, logits=logits
            )
        return loss

    def build_predictions(self, inputs_ph):
        """Builds the model loss on top provided data placeholders."""
        with tf.name_scope(self.name):
            # Build logits.
            # <float32> [None, num_classes].
            logits = self.build_logits(inputs_ph)
            # Build predictions.
            # <float32> [None].
            preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return preds


class FeedForwardModel(ClassificationModel):
    """Feed-forward classification model.

    Parameters
    ----------
    input_shapes : TensorShape
        The shape of the input tensors expected by the model.

    input_types : TensorType
        The type of the input tensors expected by the model.

    num_classes : int
        The number of classes.

    network_builder : function
        Builds the underlying network and returns a tf.keras.Model.

    name : str, optional
        Model name.
    """

    def __init__(
        self,
        input_shapes,
        input_types,
        num_classes,
        network_builder,
        name=None,
        **_unused_kwargs
    ):
        super(FeedForwardModel, self).__init__(
            input_shapes=input_shapes,
            input_types=input_types,
            name=(name or self.__class__.__name__),
        )
        self.num_classes = num_classes
        self.network_builder = network_builder

        # Internals.
        self.network = None

    def _build(self):
        """Builds all parametric components of the model graph."""
        self.network = self.network_builder(
            output_size=self.num_classes,
            input_shape=self.input_shapes,
            input_type=self.input_types,
        )
        if self.initial_parameters is None:
            self.initial_parameters = self.network.trainable_variables

    def _build_logits(self, inputs_ph):
        """Builds a part of the model graph for computing output logits."""
        # Note: `training=True` is required for correct functioning of batch
        #        normalization in meta-learning to make sure it does not use
        #        accumulated 1st and 2nd moments that may differ between tasks.
        # <float32> [None, num_classes].
        logits = self.network(inputs_ph, training=True)
        return logits

    @property
    def parameters(self):
        """Returns a list of variables that parameterize the model."""
        return self.network.variables

    @property
    def trainable_parameters(self):
        """Returns a list with the subset of variables that are trainable."""
        return self.network.trainable_variables


class ProtoModel(ClassificationModel):
    """Prototype-based classification model.

    Parameters
    ----------
    input_shapes : TensorShape
        The shape of the input tensors expected by the model.

    input_types : TensorType
        The type of the input tensors expected by the model.

    embedding_dim : int
        The size of the input embeddings used by the model.

    network_builder : function
        Builds the underlying network and returns a tf.keras.Model.

    name : str, optional
    """

    def __init__(
        self,
        input_shapes,
        input_types,
        embedding_dim,
        network_builder,
        name=None,
        **_unused_kwargs
    ):
        super(ProtoModel, self).__init__(
            input_shapes=input_shapes,
            input_types=input_types,
            name=(name or self.__class__.__name__),
        )
        self.embedding_dim = embedding_dim
        self.network_builder = network_builder

        # Internals.
        self.network = None
        self.prototypes = None

    def _build(self):
        """Builds all parametric components of the model graph."""
        self.network = self.network_builder(
            output_size=self.embedding_dim,
            input_shape=self.input_shapes,
            input_type=self.input_types,
        )
        if self.initial_parameters is None:
            self.initial_parameters = self.network.trainable_variables

    def _build_logits(self, inputs_ph, training=True):
        """Builds logits using distances between inputs and prototypes."""
        # Note: `training=True` is required for correct functioning of batch
        #        normalization in meta-learning to make sure it does not use
        #        accumulated 1st and 2nd moments that may differ between tasks.
        # <float32> [None, embedding_dim].
        embeddings = self.network(inputs_ph, training=training)

        # TODO: generalize to other types of distances.
        # <float32> [None, num_classes].
        square_dists_emb_proto = tf.reduce_sum(
            tf.square(
                # <float32> [None, 1, emb_size].
                tf.expand_dims(embeddings, 1)
                -
                # <float32> [1, num_classes, emb_size].
                tf.expand_dims(self.prototypes, 0)
            ),
            axis=-1,
        )
        # <float32> [None, num_classes].
        return tf.nn.log_softmax(-square_dists_emb_proto, axis=-1)

    @property
    def parameters(self):
        """Returns a list of variables that parameterize the model."""
        return self.network.variables

    @property
    def trainable_parameters(self):
        """Returns a list with the subset of variables that are trainable."""
        return self.network.trainable_variables
