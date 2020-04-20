from meta_blocks.models import classification


def get(name, **kwargs):
    if name == "feed_forward":
        model = classification.FeedForwardModel(**kwargs)
    elif name == "proto":
        model = classification.ProtoModel(**kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return model
