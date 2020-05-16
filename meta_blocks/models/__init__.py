import functools

from meta_blocks.models import classification


def get(name, **kwargs):
    if name == "feed_forward":
        model_builder = functools.partial(classification.FeedForwardModel, **kwargs)
    elif name == "proto":
        model_builder = functools.partial(classification.ProtoModel, **kwargs)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model_builder
