"""Simple networks."""

from typing import Any, Dict, List, Optional, Tuple, Union

import tensorflow.compat.v1 as tf

__all__ = ["build_mlp", "build_convnet"]

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


def build_mlp(
    input_shape: Tuple[int],
    input_type: str,
    *,
    output_size: Optional[int] = None,
    hidden_sizes: Tuple[int],
    activation: str = "relu",
    batch_norm: Optional[Dict[str, Any]] = None,
    output_activation: Optional[str] = None,
    name: str = "SimpleMLP",
    **_unused_kwargs,
):
    """Builds a simple MLP network.

    Allows to define a stack of identical fully connected layers optionally
    interleaved with batch normalization, and nonlinear activations. The final
    layers is fully connected with `output_size` units.
    """
    # Input node.
    inputs = tf.keras.layers.Input(shape=input_shape, dtype=input_type)
    # Build stack of layers.
    x = tf.keras.layers.Flatten()(inputs)
    for i, hidden_size in enumerate(hidden_sizes):
        x = tf.keras.layers.Dense(units=hidden_size, name=f"fc{i}")(x)
        # Batch norm (optional).
        if batch_norm is not None:
            x = tf.keras.layers.BatchNormalization(**batch_norm, name=f"bn{i}")(x)
        # Activation.
        x = tf.keras.layers.Activation(activation, name=f"activation{i}")(x)
    # Add a fully connected output layer.
    if output_size is not None:
        OutputLayer = tf.keras.layers.Dense(
            output_size, activation=output_activation, name="fc"
        )
        x = OutputLayer(x)
    # Build and output the model.
    model = tf.keras.models.Model(inputs, x, name=name)
    return model


def build_convnet(
    input_shape: Tuple[int],
    input_type: str,
    *,
    output_size: Optional[int] = None,
    filters: List[int],
    kernel_size: Union[int, Tuple[int]],
    conv2d_kwargs: Optional[Dict[str, Any]] = None,
    activation: str = "relu",
    pooling: Optional[str] = None,
    pooling_kwargs: Optional[Dict[str, Any]] = None,
    batch_norm: Optional[Dict[str, Any]] = None,
    output_activation: Optional[str] = None,
    name: str = "SimpleConvNet",
    **_unused_kwargs,
):
    """Builds a simple convolutional network.

    Allows to define a stack of identical convolutional layers optionally
    interleaved with batch normalization, activations, and pooling. The final
    layers is fully connected with `output_size` units.
    """
    # Input node.
    inputs = tf.keras.layers.Input(shape=input_shape, dtype=input_type)
    # Build stack of layers.
    x = inputs
    for i in range(len(filters)):
        x = tf.keras.layers.Conv2D(
            filters=filters[i],
            kernel_size=kernel_size,
            **conv2d_kwargs,
            name=f"conv{i}",
        )(x)
        # Batch norm (optional).
        if batch_norm is not None:
            x = tf.keras.layers.BatchNormalization(**batch_norm, name=f"bn{i}")(x)
        # Activation.
        x = tf.keras.layers.Activation(activation, name=f"activation{i}")(x)
        # Pooling (optional).
        if pooling == "avg":
            PoolingLayer = tf.keras.layers.AvgPool2D(
                name=f"avgpool{i}", **(pooling_kwargs or {})
            )
            x = PoolingLayer(x)
        elif pooling == "max":
            PoolingLayer = tf.keras.layers.MaxPool2D(
                name=f"maxpool{i}", **(pooling_kwargs or {})
            )
            x = PoolingLayer(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    # Add a fully connected layer, if necessary.
    if output_size is not None:
        OutputLayer = tf.keras.layers.Dense(
            output_size, activation=output_activation, name="fc"
        )
        x = OutputLayer(x)
    # Build and output the model.
    model = tf.keras.models.Model(inputs, x, name=name)
    return model
