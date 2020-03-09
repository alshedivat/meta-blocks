from meta_blocks.models import base
from meta_blocks.models import omniglot

from meta_blocks.models.base import Model


def get(name, dataset_name, build=True, **kwargs):
    if dataset_name == "omniglot":
        model = omniglot.get(name, build=build, **kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return model
