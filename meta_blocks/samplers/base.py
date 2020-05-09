"""Base classes and functionality for sampling."""

import abc
from typing import Optional, Tuple

import numpy as np
import tensorflow.compat.v1 as tf

# Transition to V2 will happen in stages.
tf.disable_v2_behavior()
tf.enable_resource_variables()


class Sampler(abc.ABC):
    """Abstract class for sampling instances to be labeled."""

    stateful = False

    def __init__(self, stratified: bool = False, name: Optional[str] = None):
        """Instantiates a Sampler.

        Parameters
        ----------
        stratified : bool, optional (default=False)
            If set to True, use stratification.

        name : str, optional (default="Sampler")
            The name of the sampler.
        """
        self.stratified = stratified
        self.name = name

        self._built = False

    # --- Methods. ---

    def build(self, *args, **kwargs):
        """Builds sampler internals."""
        if not self._built:
            with tf.name_scope(self.name):
                self._build(*args, **kwargs)
            self._built = True
        return self

    # --- Abstract methods. ---

    @abc.abstractmethod
    def _build(self, *args, **kwargs):
        """Builds sampler internals. Must be implemented in a subclass."""

    @abc.abstractmethod
    def select_labeled(self, size: int, **kwargs) -> Tuple[np.ndarray]:
        """Return indices of the selected labeled support points.
        Must be implemented in a subclass.
        """
