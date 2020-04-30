from meta_blocks.datasets import omniglot
from meta_blocks.datasets.base import *


def get_data_source(name, **kwargs):
    if name == "omniglot":
        data_source = omniglot.OmniglotDataSource(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}.")
    return data_source


def get_meta_dataset(name, **kwargs):
    if name == "omniglot":
        meta_dataset = omniglot.OmniglotMetaDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}.")
    return meta_dataset
