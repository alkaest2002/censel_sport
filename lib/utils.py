# mypy: disable-error-code="misc"

from itertools import count
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from . import MT100, MT1000, PUSHUPS, SITUPS, SWIM25


def generate_synthetic_data(metric_type: str, n_samples: int = 500) -> Any:
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
    base_synthetic_data = {
        SWIM25: {
            "distributions": [
                (np.random.normal, (11.5, 0.8), 0.05),
                (np.random.normal, (14.0, 1.0), 0.15),
                (np.random.normal, (18.5, 2.5), 0.60),
                (np.random.normal, (25.0, 3.0), 0.20),
            ],
            "bounds": (8.0, 40.0),
            "discrete": False,
            "unit": "seconds",
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
            "unit": "seconds",
        },
        MT1000: {
            "distributions": [
                (np.random.normal, (150, 12), 0.05),  # ~2.5 minutes
                (np.random.normal, (192, 18), 0.15),  # ~3.2 minutes
                (np.random.normal, (270, 48), 0.60),  # ~4.5 minutes
                (np.random.normal, (390, 60), 0.20),  # ~6.5 minutes
            ],
            "bounds": (120, 600),  # 2-10 minutes
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
            "unit": "counts",
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
            "unit": "counts",
        },
    }

    # Check if metric type is known
    if metric_type not in base_synthetic_data:
        raise ValueError(f"Unknown metric_type {metric_type} for synthetic data generation.")

    # Get configuration for the specified metric type
    config: dict[str, Any] = base_synthetic_data[metric_type]

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


def apply_standardization(
        data_to_standardize: NDArray[np.integer[Any] | np.floating[Any]],
        cutoffs: list[tuple],
        higher_is_better: bool = False,
    ) -> list[dict[str, float]]:
    """
    Standardize data with percentile cutoffs.

    Parameters:
    -----------
    data_to_standardize : np.array
        Data to be standardized

    cutoffs : list of tuples
        List of tuples containing percentile cutoffs

    higher_is_better : bool
        Whether higher values indicate better performance

    Returns:
    --------
    list : Original data and Standardized data
    """

    # Convert data to pandas Series for easier manipulation
    data_series = pd.Series(data_to_standardize)

    # Add inclusive bounds to cutoffs
    # First cutoff is inclusive on both sides, others only on the right
    cutoffs_with_inclusive = zip(cutoffs, ["both", *["right"] * (len(cutoffs) - 1)], strict=True)

    # initialize counter based on whether higher values denotes better performance
    counter = count(start=1, step=1) if higher_is_better else count(start=len(cutoffs), step=-1)

    # Compute standardized data
    standardized_data = data_series.case_when(
        [
            (lambda x, cutoffs=cutoffs, inclusive=inclusive:
                x.between(cutoffs[0], cutoffs[1], inclusive=inclusive), next(counter))
            for (cutoffs, inclusive) in cutoffs_with_inclusive
        ],
    )

    # zip original values with standardized values
    zipped_data = zip(data_series.tolist(), standardized_data.tolist(), strict=True)

    return [
        {
            "original_value": original_value,
            "standardized_value": standardized_value,
        }
        for original_value, standardized_value in zipped_data
    ]


def is_falsy(value: Any) -> bool:
    """
    Check if a value is falsy with support for numpy arrays and custom logic.

    Parameters
    ----------
    value : Any
        The value to check

    Returns:
    -------
    bool
        True if the value is falsy, False otherwise
    """
    # Handle None explicitly
    if value is None:
        return True

    # Handle numpy arrays
    if hasattr(value, "__len__") and hasattr(value, "size"):
        try:
            return value.size == 0 # type: ignore[no-any-return]
        except (AttributeError, TypeError):
            pass

    # Handle strings specifically (including whitespace-only strings)
    if isinstance(value, str):
        return value == "" or value.isspace()

    # Handle other containers and standard falsy values
    try:
        return len(value) == 0
    except TypeError:
        # For objects without __len__, use standard truthiness
        return not bool(value)
