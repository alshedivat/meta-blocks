"""Uniform sampling in NumPy."""

from typing import Optional, Tuple

import numpy as np

from meta_blocks.samplers import base
from meta_blocks.tasks.supervised import SupervisedTask

__all__ = ["UniformSampler"]


class UniformSampler(base.Sampler):
    """Samples instances uniformly at random."""

    stateful = False

    def __init__(self, stratified=False, name: Optional[str] = None, **_unused_kwargs):
        super(UniformSampler, self).__init__(
            stratified=stratified, name=(name or self.__class__.__name__)
        )

        # Random state must be set globally.
        self._rng = np.random

        # Internal.
        self.tasks = None

    # --- Methods. ---

    def _build(self, tasks: Tuple[SupervisedTask], **_unused_kwargs):
        """Builds a tuple of selected indices tensors."""
        self.tasks = tasks

    def select_labeled(self, size: int, **_unused_kwargs) -> Tuple[np.ndarray]:
        """Return an actively selected labeled data points from the dataset."""
        # Build selected indices tensors.
        selected_indices = []
        for i, task in enumerate(self.tasks):
            data_size = task.dataset.size
            num_classes = task.dataset.num_classes
            num_query_shots = task.num_query_shots
            # Select indices of the elements to be labeled.
            if self.stratified:
                # TODO: better handle edge cases + add tests.
                assert size % num_classes == 0
                assert data_size % num_classes == 0
                data_size_per_class = data_size // num_classes
                support_data_per_class = data_size_per_class - num_query_shots
                sample_size_per_class = size // num_classes
                # Select support elements uniformly stratified by class.
                task_indices = []
                for c in range(num_classes):
                    id_offset = c * support_data_per_class
                    c_ids = id_offset + self._rng.choice(
                        support_data_per_class,
                        size=sample_size_per_class,
                        replace=False,
                    )
                    task_indices.append(c_ids)
                task_indices = np.concatenate(task_indices)
            else:
                # Select support elements uniformly at random.
                support_data_size = data_size - (num_classes * num_query_shots)
                task_indices = self._rng.choice(
                    support_data_size, size=size, replace=False
                )
            selected_indices.append(task_indices)
        return tuple(selected_indices)
