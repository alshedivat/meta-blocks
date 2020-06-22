"""Common utility functions."""
from typing import Any, List, Tuple

import tensorflow.compat.v1 as tf

FeedList = List[Tuple[tf.Tensor, Any]]


class ModeKeys:
    """Standard names for modes."""

    TRAIN = "train"
    EVAL = "eval"
