"""A collection of backbone networks."""
import functools

from meta_blocks.networks import resnet, simple


def get(name, **kwargs):
    if name == "simple_mlp":
        network_builder = functools.partial(simple.build_mlp, **kwargs)
    elif name == "simple_cnn":
        network_builder = functools.partial(simple.build_convnet, **kwargs)
    elif name == "resnet12":
        network_builder = functools.partial(resnet.build_resnet12, **kwargs)
    else:
        raise ValueError(f"Unsupported network: {name}.")
    return network_builder
