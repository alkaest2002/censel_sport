# mypy: disable-error-code="misc"

from itertools import count
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from . import MT100, MT1000, PUSHUPS, SITUPS, SWIM25


def generate_synthetic_data(
        metric_type: str,
        n_samples: int = 300,
        random_state: int = 42) -> NDArray[np.number]:
    """Generate synthetic performance data for testing purposes.

    This function creates realistic synthetic performance data based on predefined
    distributions for different fitness metrics. The data is generated using
    multiple normal distributions to simulate real-world performance variations.

    Args:
        metric_type (str): Type of metric to generate data for. Must be one of the
            predefined metric types (SWIM25, MT100, MT1000, SITUPS, PUSHUPS).
        n_samples (int, optional): Number of samples to generate. Defaults to 300.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        NDArray[np.number]: Generated performance data as a numpy array.

    Raises:
        ValueError: If metric_type is not a recognized metric type.
    """
    rng: np.random.Generator = np.random.default_rng(random_state)

    base_synthetic_data: dict[str, dict[str, Any]] = {
        SWIM25: {
            "distributions": [
                (rng.normal, (11.5, 0.8), 0.05),
                (rng.normal, (14.0, 1.0), 0.15),
                (rng.normal, (18.5, 2.5), 0.60),
                (rng.normal, (25.0, 3.0), 0.20),
            ],
            "bounds": (8.0, 40.0),
            "discrete": False,
            "unit": "seconds",
        },
        MT100: {
            "distributions": [
                (rng.normal, (10.0, 0.3), 0.05),
                (rng.normal, (11.5, 0.5), 0.15),
                (rng.normal, (13.5, 1.0), 0.60),
                (rng.normal, (16.5, 1.5), 0.20),
            ],
            "bounds": (8.0, 25.0),
            "discrete": False,
            "unit": "seconds",
        },
        MT1000: {
            "distributions": [
                (rng.normal, (150, 12), 0.05),  # ~2.5 minutes
                (rng.normal, (192, 18), 0.15),  # ~3.2 minutes
                (rng.normal, (270, 48), 0.60),  # ~4.5 minutes
                (rng.normal, (390, 60), 0.20),  # ~6.5 minutes
            ],
            "bounds": (120, 600),  # 2-10 minutes
            "discrete": False,
        },
        SITUPS: {
            "distributions": [
                (rng.normal, (60, 8), 0.05),
                (rng.normal, (45, 6), 0.15),
                (rng.normal, (30, 8), 0.60),
                (rng.normal, (15, 5), 0.20),
            ],
            "bounds": (0, None),
            "discrete": True,
            "unit": "counts",
        },
        PUSHUPS: {
            "distributions": [
                (rng.normal, (75, 10), 0.05),
                (rng.normal, (55, 8), 0.15),
                (rng.normal, (30, 10), 0.60),
                (rng.normal, (12, 4), 0.20),
            ],
            "bounds": (0, None),
            "discrete": True,
            "unit": "counts",
        },
    }

    # Check if metric type is known
    if metric_type not in base_synthetic_data:
        raise ValueError(f"---> Unknown metric_type {metric_type} for synthetic data generation.")

    # Get configuration for the specified metric type
    config: dict[str, Any] = base_synthetic_data[metric_type]

    # Generate data from multiple distributions
    data_parts: list[NDArray[np.number[Any]]] = []
    for func, params, proportion in config["distributions"]:
        size: int = int(n_samples * proportion)
        data_parts.append(func(*params, size))

    # Concatenate data
    data: NDArray[np.number[Any]] = np.concatenate(data_parts)

    # Apply bounds
    lower: float | None
    upper: float | None
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
        data_to_standardize: NDArray[np.number[Any]],
        cutoffs: list[tuple[float, float]],
        higher_is_better: bool = False,
    ) -> pd.DataFrame:
    """Apply standardization to data using percentile cutoffs.

    This function standardizes data by assigning scores based on percentile ranges.
    The scoring direction depends on whether higher values indicate better performance.
    Each data point is assigned a standardized score corresponding to its percentile
    range.

    Args:
        data_to_standardize (NDArray[np.number[Any]]): Numerical data array to be
            standardized.
        cutoffs (list[tuple[float, float]]): List of tuples containing percentile
            cutoff ranges. Each tuple should contain (lower_bound, upper_bound) for
            a score category.
        higher_is_better (bool, optional): Whether higher values indicate better
            performance. If True, higher scores are assigned to better performance
            ranges. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with three columns:
            - original_value: The original data values
            - standardized_value: The assigned standardized scores
            - standardized_value_bounds: The cutoff ranges for each score category.
    """
    # Convert data to pandas Series for easier manipulation
    data: pd.Series = pd.Series(data_to_standardize)

    # Add inclusive bounds to cutoffs
    # First cutoff is inclusive on both sides, others only on the right
    cutoffs_with_inclusive: list[tuple[tuple[float, float], str]] = \
        list(zip(cutoffs, ["both", *["right"] * (len(cutoffs) - 1)], strict=True))

    # Initialize counter based on whether higher values denotes better performance
    counter: count = count(start=1, step=1) if higher_is_better else count(start=len(cutoffs), step=-1)

    # Compute standardized scores
    standardized_scores: pd.Series = data.case_when(
        [
            (lambda x, cutoffs=cutoffs, inclusive=inclusive:
                x.between(cutoffs[0], cutoffs[1], inclusive=inclusive), next(counter))
            for (cutoffs, inclusive) in cutoffs_with_inclusive
        ],
    )

    # Compute standardized scores lower bounds for data
    standardized_bounds: pd.Series = data.case_when(
        [
            (lambda x, cutoffs=cutoffs, inclusive=inclusive:
                x.between(cutoffs[0], cutoffs[1], inclusive=inclusive), f"{cutoffs[0]} - {cutoffs[1]}")
            for (cutoffs, inclusive) in cutoffs_with_inclusive
        ],
    )

    return pd.concat(
        [
            data,
            standardized_scores,
            standardized_bounds,
        ],
        keys=["original_value", "standardized_value", "standardized_value_bounds"],
        axis=1,
    )


def compute_sample_size(
    data_dict: dict[str, Any],
) -> int:
    """Compute sample size for statistical analysis.

    This function computes the appropriate sample size by taking the minimum
    value among bootstrap replicate size, Monte Carlo size, and median sample
    size from database groups. It queries the database to get group sizes
    and returns the most conservative estimate.

    Args:
        data_dict (dict[str, Any]): Dictionary containing configuration and data.
            Expected keys include:
            - metric_config: Configuration for database query
            - clean: Dictionary containing cleaned data array
            - bootstrap_sample_size: Bootstrap replication size
            - montecarlo_sample_size: Monte Carlo simulation size

    Returns:
        int: Computed sample size as the minimum of available size measures.
    """
    # Extract data from dictionary
    query_from_db: pd.DataFrame = data_dict.get("query_from_db", pd.DataFrame())
    clean: dict[str, Any] = data_dict.get("clean", {})
    data: NDArray[np.number[Any]] = clean.get("data", np.array([]))
    metric_config: dict[str, Any] = data_dict.get("metric_config", {})
    fixed_sample_size: int | None = metric_config.get("fixed_sample_size")
    bootstrap_sample_size: int | None = metric_config.get("bootstrap_sample_size")

    # If fixed sample size is provided, use it
    if fixed_sample_size is not None:
        return fixed_sample_size

    # if bootstrap sampe size is set, use it
    if bootstrap_sample_size is not None:
        return bootstrap_sample_size

    # If no data in DB, use data size
    if query_from_db.empty:
        sample_size_from_db: float = data.size
    else:
        sample_size_from_db = query_from_db.groupby(["recruitment_year", "recruitment_type"]).size().median()

    # Compute sample size as the minimum of the three sizes
    sample_size: int  = int(sample_size_from_db)

    # The final number should be rounded down to the nearest 50
    # Examples: 274 -> 300, 225 -> 200
    return  math.floor(sample_size / 50) * 50

