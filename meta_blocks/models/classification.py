"""Base models for meta-learning."""

import abc
import logging
from typing import Callable, List, Optional

import tensorflow.compat.v1 as tf

from meta_blocks import networks
from meta_blocks.models.base import Model

__all__ = ["ClassificationModel"]

logger = logging.getLogger(__name__)

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


class ClassificationModel(abc.ABC, Model):
    """Abstract base class for classification models.

    Subclasses must implement ``_build_network``, ``_build_embeddings``, and
    ``_build_logits``.

    Parameters
    ----------
    input_shapes : TensorShape
        The shape of the input tensors expected by the model.

    input_types : DType
        The type of the input tensors expected by the model.

    name : str, optional
        The name of the model. Used to define the corresponding name scope.
    """

    def __init__(
        self, input_shapes: tf.TensorShape, input_types: tf.DType, name: Optional[str]
    ):
        self.input_shapes = input_shapes
        self.input_types = input_types
        self.name = name or self.__class__.__name__

        # Build the model graph.
        with tf.name_scope(self.name):
            self._build_network()

    # --- Abstract properties. ---

    @property
    @abc.abstractmethod
    def parameters(self) -> List[tf.Tensor]:
        """Returns a list of variables that parameterize the model.
        Must be implemented in a subclass.
        """

    @property
    @abc.abstractmethod
    def trainable_parameters(self) -> List[tf.Tensor]:
        """Returns a list with the subset of variables that are trainable.
        Must be implemented in a subclass.
        """

    @property
    @abc.abstractmethod
    def non_trainable_parameters(self) -> List[tf.Tensor]:
        """Returns a list with the subset of variables that are non-trainable.
        Must be implemented in a subclass.
        """

    # --- Methods. ---

    def embeddings(self, inputs: tf.Tensor) -> tf.Tensor:
        """Builds embeddings for the provided inputs."""
        with tf.name_scope(self.name):
            # Build embeddings: <float32> [None, embedding_dim].
            embeddings = self._build_embeddings(inputs)
        return embeddings

    def logits(self, inputs: tf.Tensor) -> tf.Tensor:
        """Builds the logits for the provided inputs."""
        with tf.name_scope(self.name):
            # Build logits: <float32> [None, num_classes].
            logits = self._build_logits(inputs)
        return logits

    def loss(self, inputs: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        """Builds the model loss on top provided data tensors."""
        with tf.name_scope(self.name):
            # Build logits: <float32> [None, num_classes].
            logits = self._build_logits(inputs)
            # Build loss: <float32> [].
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits
                )
            )
        return loss

    def predictions(self, inputs: tf.Tensor) -> tf.Tensor:
        """Builds the model predictions for the provided input tensors."""
        with tf.name_scope(self.name):
            # Build logits: <float32> [None, num_classes].
            logits = self._build_logits(inputs)
            # Build preds_and_labels: <float32> [None].
            preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return preds

    # --- Abstract methods. ---

    @abc.abstractmethod
    def _build_network(self) -> None:
        """Builds the model graph for prediction.
        Must be implemented in a subclass.
        """

    @abc.abstractmethod
    def _build_embeddings(self, inputs: tf.Tensor) -> tf.Tensor:
        """Builds a part of the model graph for computing input embeddings.
        Must be implemented in a subclass.
        """

    @abc.abstractmethod
    def _build_logits(self, inputs: tf.Tensor) -> tf.Tensor:
        """Builds a part of the model graph for computing output logits.
        Must be implemented in a subclass.
        """


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
        If ``embedding_dim`` is not None, it is used to build the body network.

    embedding_dim : int, optional
        If provided, builds a 2-part network that consists of the body and head.
        The body is built by the ``network_builder`` and computes embeddings.
        The head is built on top of embeddings by the ``head_network_builder``
        and computes the logits. This enables one-way compatibility between
        feed-forward and proto models (i.e., a network trained under a feed-
        forward model can be used as a building block of a proto model at
        evaluation time.

    head_network_builder : function, optional
        Builds a network for computing logits from embeddings a tf.keras.Model.
        Only used if ``embedding_dim`` is not None. Defaults to logistic
        regression model (i.e., a single Dense layer).

    name : str, optional
        Model name.
    """

    def __init__(
        self,
        input_shapes: tf.TensorShape,
        input_types: tf.DType,
        num_classes: int,
        network_builder: Callable,
        embedding_dim: Optional[int] = None,
        head_network_builder: Optional[Callable] = None,
        name: Optional[str] = None,
        **_unused_kwargs
    ):
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.network_builder = network_builder
        self.head_network_builder = head_network_builder

        # Internals.
        self.network = None
        self._body_network = None
        self._head_network = None

        super(FeedForwardModel, self).__init__(
            input_shapes=input_shapes,
            input_types=input_types,
            name=(name or self.__class__.__name__),
        )

    # --- Properties. ---

    @property
    def parameters(self) -> List[tf.Tensor]:
        """Returns a list of variables that parameterize the model."""
        return self.network.variables

    @property
    def trainable_parameters(self) -> List[tf.Tensor]:
        """Returns a list with the subset of variables that are trainable."""
        return self.network.trainable_variables

    @property
    def non_trainable_parameters(self) -> List[tf.Tensor]:
        """Returns a list with the subset of variables that are non-trainable."""
        return self.network.non_trainable_variables

    # --- Methods. ---

    def _build_network(self) -> None:
        """Builds all parametric components of the model graph."""
        if self.embedding_dim is not None:
            # Get a head network builder, if not provided.
            if self.head_network_builder is None:
                self.head_network_builder = networks.get(
                    name="simple_mlp", hidden_sizes=[]
                )
            # Build body and head networks.
            self._body_network = self.network_builder(
                output_size=self.embedding_dim,
                input_shape=self.input_shapes,
                input_type=self.input_types,
            )
            self._head_network = self.head_network_builder(
                output_size=self.num_classes,
                input_shape=self._body_network.outputs[0].shape[1:],
                input_type=self._body_network.outputs[0].dtype,
            )
            # Compose networks.
            self.network = tf.keras.Model(
                inputs=self._body_network.inputs,
                outputs=self._head_network(self._body_network.outputs),
            )
        else:
            self.network = self.network_builder(
                output_size=self.num_classes,
                input_shape=self.input_shapes,
                input_type=self.input_types,
            )

    def _build_embeddings(self, inputs: tf.Tensor) -> tf.Tensor:
        """Builds a part of the model graph for computing input embeddings."""
        embeddings = None
        if self._body_network is not None:
            # Build embeddings: <float32> [None, embedding_dim].
            embeddings = self._body_network(inputs)
        return embeddings

    def _build_logits(self, inputs: tf.Tensor) -> tf.Tensor:
        """Builds a part of the model graph for computing output logits."""
        # Build logits: <float32> [None, num_classes].
        return self.network(inputs)


class ProtoModel(ClassificationModel):
    """Prototype-based classification model.

    Parameters
    ----------
    input_shapes : TensorShape
        The shape of the input tensors expected by the model.

    input_types : TensorType
        The type of the input tensors expected by the model.

    network_builder : function
        Builds the underlying network and returns a tf.keras.Model.

    embedding_dim : int
        The size of the input embeddings used by the model.

    network_builder : function
        Builds the underlying network and returns a tf.keras.Model.

    name : str, optional
    """

    def __init__(
        self,
        input_shapes: tf.TensorShape,
        input_types: tf.DType,
        network_builder: Callable,
        embedding_dim: Optional[int] = None,
        name: Optional[str] = None,
        **_unused_kwargs
    ):
        self.embedding_dim = embedding_dim
        self.network_builder = network_builder

        # Internals.
        self.network = None
        self.prototypes = None

        super(ProtoModel, self).__init__(
            input_shapes=input_shapes,
            input_types=input_types,
            name=(name or self.__class__.__name__),
        )

    # --- Properties. ---

    @property
    def parameters(self) -> List[tf.Tensor]:
        """Returns a list of variables that parameterize the model."""
        return self.network.variables

    @property
    def trainable_parameters(self) -> List[tf.Tensor]:
        """Returns a list with the subset of variables that are trainable."""
        return self.network.trainable_variables

    @property
    def non_trainable_parameters(self) -> List[tf.Tensor]:
        """Returns a list with the subset of variables that are non-trainable."""
        return self.network.non_trainable_variables

    # --- Methods. ---

    def logits(self, inputs: tf.Tensor) -> Optional[tf.Tensor]:
        """Builds logits for the provided input tensors.
        If model's prototypes are undefined, returns None.
        """
        if self.prototypes is None:
            return None
        return super(ProtoModel, self).logits(inputs)

    def loss(self, inputs: tf.Tensor, labels: tf.Tensor) -> Optional[tf.Tensor]:
        """Builds the model loss on top provided data tensors.
        If model's prototypes are undefined, returns None.
        """
        if self.prototypes is None:
            return None
        return super(ProtoModel, self).loss(inputs, labels)

    def predictions(self, inputs: tf.Tensor) -> tf.Tensor:
        """Builds the model predictions for the provided input tensors.
        If model's prototypes are undefined, returns None.
        """
        if self.prototypes is None:
            return None
        return super(ProtoModel, self).predictions(inputs)

    def _build_network(self) -> None:
        """Builds all parametric components of the model graph."""
        self.network = self.network_builder(
            output_size=self.embedding_dim,
            input_shape=self.input_shapes,
            input_type=self.input_types,
        )
        if self.embedding_dim is None:
            self.embedding_dim = self.network.output.shape[-1]

    def _build_embeddings(self, inputs: tf.Tensor) -> tf.Tensor:
        """Builds a part of the model graph for computing input embeddings."""
        # Build embeddings: <float32> [None, embedding_dim].
        return self.network(inputs)

    def _build_logits(self, inputs: tf.Tensor) -> tf.Tensor:
        """Builds logits using distances between inputs and prototypes."""
        # Build embeddings: <float32> [None, embedding_dim].
        embeddings = self._build_embeddings(inputs)
        # TODO: generalize to other types of distances.
        # Build pairwise L2 distances: <float32> [None, num_classes].
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
        # Build logits: <float32> [None, num_classes].
        return tf.nn.log_softmax(-square_dists_emb_proto, axis=-1)
