"""Utility functions for meta-datasets."""

from typing import List, Optional, Tuple

import numpy as np

from meta_blocks.datasets.base import DataSource

# Types.
DatasetRequest = Tuple[
    # Data source IDs that represent dataset classes.
    np.ndarray,
    # A tuple of selected image ids for each data class.
    Tuple[np.ndarray],
]


def generate_dataset_request(
    data_sources: List[DataSource],
    num_classes: int,
    unique_classes: bool,
    data_source_size: Optional[int] = None,
    rng: Optional[np.random.RandomState] = None,
) -> DatasetRequest:
    """Generates a dataset request."""
    rng = rng or np.random
    # Sample data sources.
    data_source_ids = rng.choice(
        len(data_sources), size=num_classes, replace=(not unique_classes)
    )
    # Sample selected image ids for each data source.
    selected_ids = tuple(
        rng.choice(
            data_sources[i].size,
            size=data_source_size or data_sources[i].size,
            replace=False,
        )
        for i in data_source_ids
    )
    return data_source_ids, selected_ids
