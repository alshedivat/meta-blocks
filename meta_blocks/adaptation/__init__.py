from meta_blocks.adaptation.base import *
from meta_blocks.adaptation.maml import *
from meta_blocks.adaptation.proto import *
from meta_blocks.adaptation.reptile import *


def get(name, build=True, **kwargs):
    if name == "maml":
        meta_learner = Maml(**kwargs)
    elif name == "proto":
        meta_learner = Proto(**kwargs)
    elif name == "reptile":
        meta_learner = Reptile(**kwargs)
    else:
        raise ValueError(f"Unknown adaptation strategy: {name}.")
    if build:
        meta_learner.build()
    return meta_learner
