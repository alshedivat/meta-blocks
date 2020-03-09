from meta_blocks.datasets.base import *
from meta_blocks.datasets import omniglot_v2 as omniglot


def get_categories(name, data_dir, **kwargs):
    if name == "omniglot":
        categories = omniglot.get_categories(data_dir, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}.")
    return categories


def get_datapool(dataset_name, **kwargs):
    if dataset_name == "omniglot":
        data_pool = omniglot.OmniglotDataPool(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}.")
    return data_pool


def get_metadataset(dataset_name, **kwargs):
    if dataset_name == "omniglot":
        meta_dataset = omniglot.OmniglotMetaDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}.")
    return meta_dataset
