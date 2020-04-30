from meta_blocks.samplers import uniform
from meta_blocks.samplers.base import Sampler


def get(name, **kwargs):
    """Returns a sampling strategy instance for the given model."""
    if name == "uniform":
        sampler = uniform.UniformSampler(**kwargs)
    else:
        raise ValueError(f"Unknown sampler: {name}")
    return sampler
