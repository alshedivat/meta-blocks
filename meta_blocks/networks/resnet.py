"""Resnets."""

from typing import Optional, Tuple

import tensorflow.compat.v1 as tf
from tensorflow.python.keras.applications import resnet

__all__ = ["build_resnet12"]

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


def ResNet12(input_tensor=None, input_shape=None, pooling=None, classes=1000, **kwargs):
    """Instantiates the ResNet12 architecture."""

    def stack_fn(x):
        x = resnet.stack1(x, 64, 1, stride1=1, name="conv2")
        x = resnet.stack1(x, 128, 1, name="conv3")
        x = resnet.stack1(x, 256, 1, name="conv4")
        return resnet.stack1(x, 512, 1, name="conv5")

    return resnet.ResNet(
        stack_fn=stack_fn,
        preact=False,
        use_bias=True,
        model_name="resnet50",
        include_top=False,
        weights=None,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        **kwargs,
    )


def build_resnet12(
    input_shape: Tuple[int],
    input_type: str,
    *,
    pooling: Optional[str] = None,
    output_size: Optional[int] = None,
    output_activation: Optional[str] = None,
    name: str = "ResNet12",
):
    """Builds the standard ResNet12 network."""
    # Input node.
    inputs = tf.keras.layers.Input(shape=input_shape, dtype=input_type)
    # Transform inputs with ResNet12.
    x = ResNet12(input_shape=input_shape, pooling=pooling)(inputs)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    # Add a fully connected output layer.
    if output_size is not None:
        OutputLayer = tf.keras.layers.Dense(
            output_size, activation=output_activation, name="fc"
        )
        x = OutputLayer(x)
    # Build and output the model.
    model = tf.keras.models.Model(inputs, x, name=name)
    return model
