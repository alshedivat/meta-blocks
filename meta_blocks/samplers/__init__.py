from meta_blocks.samplers.base import Sampler
from meta_blocks.samplers.uniform import UniformSampler


def get(name, build=True, **kwargs):
    """Returns a sampling strategy instance for the given model."""
    if name is None:
        return None
    if name == "uniform":
        sampler = UniformSampler(**kwargs)
    else:
        print(f"Unknown sampler: {name}")
        raise NotImplementedError
    if build:
        sampler.build(**kwargs)
    return sampler
