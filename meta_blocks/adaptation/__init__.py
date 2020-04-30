from meta_blocks.adaptation.base import AdaptationStrategy
from meta_blocks.adaptation.maml import Maml
from meta_blocks.adaptation.proto import Proto
from meta_blocks.adaptation.reptile import Reptile


def get(name, **kwargs):
    if name == "maml":
        meta_learner = Maml(**kwargs)
    elif name == "proto":
        meta_learner = Proto(**kwargs)
    elif name == "reptile":
        meta_learner = Reptile(**kwargs)
    else:
        raise ValueError(f"Unknown adaptation strategy: {name}.")
    return meta_learner
