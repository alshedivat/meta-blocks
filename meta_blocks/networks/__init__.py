"""A collection of backbone networks."""
import functools

from meta_blocks.networks import simple


def get(name, **kwargs):
    if name == "simple_mlp":
        network_builder = functools.partial(simple.build_mlp, **kwargs)
    elif name == "simple_cnn":
        network_builder = functools.partial(simple.build_convnet, **kwargs)
    else:
        raise ValueError(f"Unsupported network: {name}.")
    return network_builder
