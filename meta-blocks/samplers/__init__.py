from .base import *
from .cluster import *
from .margin import *
from .uniform import *


def get(name, build=True, **kwargs):
    """Returns a sampling strategy instance for the given model."""
    if name is None:
        return None
    if name == "uniform":
        sampler = UniformSampler(**kwargs)
    elif name == "cluster":
        sampler = ClusterSampler(use_margin_scores=False, **kwargs)
    elif name == "cluster_margin":
        sampler = ClusterSampler(use_margin_scores=True, **kwargs)
    elif name == "margin":
        sampler = MarginSampler(**kwargs)
    else:
        print(f"Unknown sampler: {name}")
        raise NotImplementedError
    if build:
        sampler.build(**kwargs)
    return sampler
