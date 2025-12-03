# mypy: disable-error-code="misc"

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
        awarded_scores: list[float],
        higher_is_better: bool = False,
    ) -> pd.DataFrame:
    """Apply standardization to data using percentile cutoffs.

    Args:
        data_to_standardize: Numerical data array to be standardized.
        cutoffs: List of tuples containing percentile cutoff ranges.
        awarded_scores: List of scores to assign for each cutoff range.
        higher_is_better: Whether higher values indicate better performance.
            If True, higher scores are assigned to better performance ranges. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with standadization results.
    """
    # Convert data to pandas Series for easier manipulation
    data: pd.Series = pd.Series(data_to_standardize)

    # Add inclusive bounds to cutoffs
    # First cutoff is inclusive on both sides, others only on the right
    cutoffs_with_inclusive: list[tuple[tuple[float, float], str]] = \
        list(zip(cutoffs, ["both", *["right"] * (len(cutoffs) - 1)], strict=True))

    # Compute standardized steps
    standardized_steps: pd.Series = data.case_when(
        [
            (lambda x, cutoffs=cutoffs, inclusive=inclusive:
                x.between(cutoffs[0], cutoffs[1], inclusive=inclusive), i)
            for i, (cutoffs, inclusive) in enumerate(cutoffs_with_inclusive, start=1)
        ],
    ).astype(int)

    # Compute standardized values based on performance direction
    standardized_values: pd.Series =\
        standardized_steps if higher_is_better else standardized_steps.rsub(len(cutoffs)+1)

    # Compute awarded scores
    mapping = dict(zip(range(1, len(cutoffs)+1), awarded_scores, strict=True))
    standardized_awarded_scores: pd.Series = standardized_steps.map(mapping).astype(float)

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
            standardized_steps,
            standardized_values,
            standardized_awarded_scores,
            standardized_bounds,
        ],
        keys=[
            "original_value",
            "standardized_step",
            "standardized_value",
            "standardized_awarded_score",
            "standardized_value_bounds"],
        axis=1,
    )


def compute_sample_size(
    data_dict: dict[str, Any],
) -> int:
    """Compute sample size for statistical analysis.

    Args:
        data_dict: Dictionary containing data.

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

    # If fixed sample size is set (in metric config), use it
    if fixed_sample_size is not None:
        return fixed_sample_size

    # if bootstrap sample size is set (via computation), use it
    if bootstrap_sample_size is not None:
        return bootstrap_sample_size

    # Compute sample size from database
    if query_from_db.empty:
        # If no data from DB, use size of cleaned data
        sample_size_from_db: float = data.size
    else:
        # Compute median sample size from recruitment year and type groups
        sample_size_from_db =(
            query_from_db
                .groupby(["recruitment_year", "recruitment_type"])
                .size()
                .quantile(.5, interpolation="nearest")
        )

    # Compute sample size as the minimum of the three sizes
    final_sample_size: int  = int(sample_size_from_db)

    # The final number should be rounded down to the nearest 50
    # Examples: 274 -> 250, 225 -> 200
    return  max(50, math.floor(final_sample_size / 50) * 50)

