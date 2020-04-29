from meta_blocks.tasks import classic_supervised, limited_supervised, self_supervised


def get_distribution(name, meta_dataset, name_suffix=None, **kwargs):
    if name == "classic_supervised":
        name_suffix = "" if name_suffix is None else f"_{name_suffix}"
        task_dist = classic_supervised.ClassicSupervisedTaskDistribution(
            meta_dataset=meta_dataset, name="CSTD" + name_suffix, **kwargs
        )
    elif name == "limited_supervised":
        name_suffix = "" if name_suffix is None else f"_{name_suffix}"
        task_dist = limited_supervised.LimitedSupervisedTaskDistribution(
            meta_dataset=meta_dataset, name="LSTD" + name_suffix, **kwargs
        )
    elif name == "self_supervised":
        name_suffix = "" if name_suffix is None else f"_{name_suffix}"
        task_dist = self_supervised.UmtraTaskDistribution(
            meta_dataset=meta_dataset, name="UMTRA" + name_suffix, **kwargs
        )
    else:
        raise ValueError(f"Unknown task distribution: {name}")
    return task_dist
