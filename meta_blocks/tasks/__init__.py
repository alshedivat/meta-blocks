from meta_blocks.tasks.base import *
from meta_blocks.tasks.supervised import *


def get_distribution(
    name, meta_dataset, build=True, name_suffix=None, **kwargs
):
    if name == "supervised":
        dist_name = "STD" + f"_{name_suffix}" if name_suffix is not None else ""
        task_dist = SupervisedTaskDistribution(
            meta_dataset=meta_dataset, name=dist_name, **kwargs
        )
    else:
        raise ValueError(f"Unknown task distribution: {name}")
    if build:
        task_dist.build()
    return task_dist
