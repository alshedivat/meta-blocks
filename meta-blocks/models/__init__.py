from . import base
from . import cob
from . import kkanji
from . import omniglot
from . import miniimagenet

from .base import Model


def get(name, dataset_name, build=True, **kwargs):
    if dataset_name == "cob":
        model = cob.get(name, build=build, **kwargs)
    elif dataset_name == "kkanji":
        model = kkanji.get(name, build=build, **kwargs)
    elif dataset_name == "omniglot":
        model = omniglot.get(name, build=build, **kwargs)
    elif dataset_name == "mini-imagenet":
        model = miniimagenet.get(name, build=build, **kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return model
