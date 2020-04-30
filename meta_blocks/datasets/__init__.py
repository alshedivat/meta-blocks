from meta_blocks.datasets import miniimagenet, omniglot


def get_data_source(name, **kwargs):
    if name == "omniglot":
        data_source = omniglot.OmniglotDataSource(**kwargs)
    elif name == "miniimagenet":
        data_source = miniimagenet.MiniImageNetDataSource(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}.")
    return data_source


def get_meta_dataset(name, **kwargs):
    if name == "omniglot":
        meta_dataset = omniglot.OmniglotMetaDataset(**kwargs)
    elif name == "miniimagenet":
        meta_dataset = miniimagenet.MiniImageNetMetaDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}.")
    return meta_dataset
