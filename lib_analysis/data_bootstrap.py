# mypy: disable-error-code="call-overload"
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.stats import iqr

from lib_analysis.utils_generic import is_falsy
from lib_analysis.utils_stats import compute_sample_size


def _compute_cutoffs(
        bootstrap_percentiles: pd.Series,
        metric_precision: int = 2,
    ) -> list[tuple[float, float]]:
    """
    Compute normative table cutoffs based on bootstrap percentiles.

    Args:
        bootstrap_percentiles:  pd.Series containing bootstrap percentiles.
        metric_precision: Decimal precision for rounding cutoffs. Defaults to 2.

    Returns:
        A list of tuples containing the normative table cutoffs as (lower_bound, upper_bound) pairs.
    """
    # Sort percentiles values and convert to list
    percentiles_values: list[float] = bootstrap_percentiles.sort_values().tolist()

    # Define cutoffs array adding 0 and a large number at the right end
    cutoffs: NDArray[np.number[Any]] = np.round([0, *percentiles_values, 1e10], metric_precision)

    # Make pairs of cutoffs in the form of: [(lower_bound, upper_bound), ...]
    cutoffs_pairs: list[tuple[float, float]] = list(pairwise(cutoffs))

    return cutoffs_pairs

def _create_percentile_statistics(
        percentiles: list[float | int],
        percentiles_estimates: list[NDArray[np.number[Any]]],
        percentile_method: str,
        ci_level: float,
    ) -> pd.DataFrame:
    """
    Create a list of percentile statistics dictionaries from bootstrap estimates.

    Args:
        percentiles: List of percentiles being analyzed.
        percentiles_estimates: List of numpy arrays containing bootstrap estimates
            for each percentile.
        percentile_method: Method used for percentile calculation (e.g., 'linear', 'nearest').
        ci_level: Confidence interval level (e.g., 0.95 for 95% CI).

    Returns:
        pandas DataFrame where each row contains statistics for a specific percentile.
    """

    # Convert boostrap percentiles estimates to DataFrame for easier aggregation
    bootstrap_df: pd.DataFrame = pd.DataFrame(percentiles_estimates, columns=percentiles)

    # Compute statistics for each percentile
    bootstrap_stats: pd.DataFrame = (pd.DataFrame({
            "percentile": bootstrap_df.columns,
            "value": bootstrap_df.quantile(0.5, interpolation=percentile_method),
            "min": bootstrap_df.min(),
            "max": bootstrap_df.max(),
            "iqr": bootstrap_df.agg(lambda x: iqr(x)),
            "first_quartile": bootstrap_df.quantile(0.25, interpolation=percentile_method),
            "third_quartile": bootstrap_df.quantile(0.75, interpolation=percentile_method),
            "ci_lower": bootstrap_df.quantile((1 - ci_level) / 2, interpolation=percentile_method),
            "ci_upper": bootstrap_df.quantile(1 - (1 - ci_level) / 2, interpolation=percentile_method),
        })
        .assign(ci_level=ci_level)
        .reset_index(drop=True)
    )

    return bootstrap_stats


def compute_bootstrap_percentiles(
    data_dict: dict[str, Any],
) -> tuple[dict[str, Any], list[NDArray[np.number[Any]]]]:
    """
    Compute bootstrap percentiles and confidence intervals from data.

    Args:
        data_dict: Dictionary containing data.

    Returns:
        A tuple containing:
            - Updated data_dict with bootstrap results added under 'bootstrap' key
            - List of bootstrap samples for further analysis

    Raises:
        ValueError: If required data (metric_config, clean, or data) is missing or falsy.
    """
    # Extract data from dictionary
    metric_config: dict[str, Any] = data_dict.get("metric_config", {})
    clean: dict[str, Any] = data_dict.get("clean", {})
    data: NDArray[np.number[Any]] = clean.get("data", np.array([]))
    requested_percentiles: list[float] = sorted(metric_config.get("requested_percentiles", [5, 25, 50, 75, 95]))
    n_replicates: int = metric_config.get("bootstrap_n_samples", 10000)
    ci_level: float = metric_config.get("bootstrap_ci_level", 0.95)
    metric_type: str | None = metric_config.get("metric_type")
    metric_precision: int = metric_config.get("metric_precision", 2)
    random_state: int = metric_config.get("random_state", 42)

    # Raise error if something crucial is missing
    if any(map(is_falsy, ( metric_config, clean, data ))):
        raise ValueError("---> The data dictionary does not contain all required parts.")

    # Initialize random generator
    rng = np.random.default_rng(random_state)

    # Comptue sample size based on data
    bootstrap_sample_size: int = compute_sample_size(data_dict)

    # Define percentile method based on type of metric (either 'continuous' or 'discrete')
    percentile_method: str = "linear" if metric_type == "continuous" else "nearest"

    # Generate all percentiles from 0 to 100 in steps of 5
    all_percentiles: list[float] = list(range(0, 101, 5))

    # Initialize lists that will hold bootstrap results
    bootstrap_samples: list[NDArray[np.number[Any]]] = []
    bootstrap_sample: NDArray[np.number[Any]] = np.array([])
    computed_all_percentiles: list[NDArray[np.number[Any]]] = []
    computed_requested_percentiles: list[NDArray[np.number[Any]]] = []

    # Bootstrap resampling
    for _ in range(n_replicates):

        # Generate bootstrap sample
        bootstrap_sample = rng.choice(data, size=bootstrap_sample_size, replace=True)

        # Store bootstrap sample for further analysis if needed
        bootstrap_samples.append(bootstrap_sample)

        # Compute all percentiles for current bootstrap sample and add to list
        computed_all_percentiles\
            .append(np.percentile(bootstrap_sample, all_percentiles, method=percentile_method))

        # Compute requested percentiles for current bootstrap sample and add to list
        computed_requested_percentiles\
            .append(np.percentile(bootstrap_sample, requested_percentiles, method=percentile_method))

    # Create list for all bootstrap percentile statistics
    all_bootstrap_percentiles: pd.DataFrame = _create_percentile_statistics(
        percentiles=all_percentiles,
        percentiles_estimates=computed_all_percentiles,
        percentile_method=percentile_method,
        ci_level=ci_level,
    )

    # Create list for requested bootstrap percentile statistics
    requested_bootstrap_percentiles: pd.DataFrame = _create_percentile_statistics(
        percentiles=requested_percentiles,
        percentiles_estimates=computed_requested_percentiles,
        percentile_method=percentile_method,
        ci_level=ci_level,
    )

    # Compute cutoffs based on requested percentiles
    percentile_cutoffs = _compute_cutoffs(requested_bootstrap_percentiles["value"], metric_precision=metric_precision)

    # Update metric config by adding bootstrap_sample_size
    data_dict["metric_config"]["bootstrap_sample_size"] = bootstrap_sample_size

    # Update data dictionary
    data_dict["bootstrap"] = {
        "all_percentiles": all_bootstrap_percentiles,
        "requested_percentiles": requested_bootstrap_percentiles,
        "cutoffs": percentile_cutoffs,
    }

    return data_dict, bootstrap_samples
