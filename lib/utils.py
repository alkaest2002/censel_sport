
from typing import Any

import numpy as np
import pandas as pd

from . import MT100, MT1000, PUSHUPS, SITUPS, SWIM25


def generate_synthetic_data(metric_type: str, n_samples: int=500) -> Any:
    """
    Generate synthetic performance data for testing purposes.

    Parameters:
    -----------
    metric_type : str
        Type of metric to generate data for
    n_samples : int
        Number of samples to generate

    Returns:
    --------
    np.array : Generated performance data
    """
    base_configs = {
        SWIM25: {
            "distributions": [
                (np.random.normal, (11.5, 0.8), 0.05),  # Elite
                (np.random.normal, (14.0, 1.0), 0.15),  # Good
                (np.random.normal, (18.5, 2.5), 0.60),  # Average
                (np.random.normal, (25.0, 3.0), 0.20),   # Poor
            ],
            "bounds": (8.0, 40.0),
            "discrete": False,
        },
        MT100: {
            "distributions": [
                (np.random.normal, (10.0, 0.3), 0.05),
                (np.random.normal, (11.5, 0.5), 0.15),
                (np.random.normal, (13.5, 1.0), 0.60),
                (np.random.normal, (16.5, 1.5), 0.20),
            ],
            "bounds": (8.0, 25.0),
            "discrete": False,
        },
        MT1000: {
            "distributions": [
                (np.random.normal, (2.5, 0.2), 0.05),
                (np.random.normal, (3.2, 0.3), 0.15),
                (np.random.normal, (4.5, 0.8), 0.60),
                (np.random.normal, (6.5, 1.0), 0.20),
            ],
            "bounds": (2.0, 10.0),
            "discrete": False,
        },
        SITUPS: {
            "distributions": [
                (np.random.normal, (60, 8), 0.05),
                (np.random.normal, (45, 6), 0.15),
                (np.random.normal, (30, 8), 0.60),
                (np.random.normal, (15, 5), 0.20),
            ],
            "bounds": (0, None),
            "discrete": True,
        },
        PUSHUPS: {
            "distributions": [
                (np.random.normal, (75, 10), 0.05),
                (np.random.normal, (55, 8), 0.15),
                (np.random.normal, (30, 10), 0.60),
                (np.random.normal, (12, 4), 0.20),
            ],
            "bounds": (0, None),
            "discrete": True,
        },
    }

    # Default config
    config: dict[str, Any] = {
        "distributions": [(np.random.normal, (50, 15), 1.0)],
        "bounds": (0, None),
        "discrete": False,
    }

    # Update config if metric_type is found
    if metric_type in base_configs:
        config = base_configs[metric_type]

    # Generate data from multiple distributions
    data_parts = []
    for func, params, proportion in config["distributions"]:
        size = int(n_samples * proportion)
        data_parts.append(func(*params, size))

    # Concatenate data
    data = np.concatenate(data_parts)

    # Apply bounds
    lower, upper = config["bounds"]
    if lower is not None:
        data = data[data > lower]
    if upper is not None:
        data = data[data < upper]

    # Make discrete if needed
    if config.get("discrete", False):
        data = np.round(data).astype(int)

    return data

def apply_standardization(data_to_standardize: np.array, cutoffs: list[tuple]) -> np.array:
    """
    Standardize data with percentile cutoffs.

    Parameters:
    -----------
    data_to_standardize : np.array
        Data to be standardized

    cutoffs : list of tuple
        List of percentile cutoffs

    Returns:
    --------
    np.array : Standardized data
    """

    # Convert data to pandas Series for easier manipulation
    data_series = pd.Series(data_to_standardize)

    # Add inclusive bounds to cutoffs
    # First cutoff is inclusive on both sides, others only on the right
    cutoffs_with_inclusive = zip(cutoffs, ["both", *["right"] * (len(cutoffs) - 1)], strict=True)

    # Compute standardized data
    standardized_data = (
       data_series
            .case_when([
                (lambda x, cutoffs=cutoffs, inclusive=inclusive:\
                    x.between(cutoffs[0], cutoffs[1], inclusive=inclusive), idx + 1)
                for idx, (cutoffs, inclusive) in enumerate(cutoffs_with_inclusive)
            ])
    )

    return standardized_data.to_numpy()
