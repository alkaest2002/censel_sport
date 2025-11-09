# mypy: disable-error-code="call-overload"
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import iqr

from lib_analysis.utils_generic import is_falsy


def _compute_cutoffs(
        bootstrap_percentiles: list[dict[str, Any]],
        metric_precision: int = 2,
    ) -> list[tuple[float, float]]:
    """
    Compute normative table cutoffs based on bootstrap percentiles.

    Parameters:
    -----------
    bootstrap_percentiles : dict
        Data dictionary with bootstrap percentile data

    metric_precision : int
        Decimal precision for rounding cutoffs

    Returns:
        list: A list containing the normative table cutoffs.
    """
    # Extract percentile values
    percentiles_values: list[int | float] = [percentile["value"] for percentile in bootstrap_percentiles]

    # Define cutoffs array
    cutoffs = np.round([0, *percentiles_values, 1e10], metric_precision)

    # Compute final cutoffs in the form of: [(lower_bound, upper_bound), ...]
    return list(pairwise(cutoffs))

def _compute_percentile_statistics(
        percentile: float,
        percentile_estimates: NDArray[np.floating[Any]],
        percentile_method: str,
        ci_level: float,
    ) -> dict[str, Any]:
    """
    Compute statistics for a given percentile from bootstrap estimates.

    Parameters:
    -----------
    percentile : int | float
        The percentile being analyzed

    percentile_estimates : NDArray
        Array of bootstrap estimates for the percentile

    percentile_method : str
        Method used for percentile calculation

    ci_level : float
        Confidence interval level

    Returns:
    --------
    dict : Dictionary containing computed statistics for the percentile
    """
    alpha: float = 1 - ci_level
    lower_ci: float = (alpha / 2) * 100
    upper_ci: float = (1 - alpha / 2) * 100

    return {
        "percentile": percentile,
        "value": np.percentile(percentile_estimates, 50, method=percentile_method),
        "min": np.min(percentile_estimates),
        "max": np.max(percentile_estimates),
        "iqr": iqr(percentile_estimates, interpolation=percentile_method),
        "first_quartile": np.percentile(percentile_estimates, 25, method=percentile_method),
        "third_quartile": np.percentile(percentile_estimates, 75, method=percentile_method),
        "ci_lower": np.percentile(percentile_estimates, lower_ci, method=percentile_method),
        "ci_upper": np.percentile(percentile_estimates, upper_ci, method=percentile_method),
        "ci_level": ci_level,
    }

def _create_percentile_statistics_list(
        percentiles: list[float | int],
        percentiles_estimates: list[NDArray[np.float64]],
        percentile_method: str,
        ci_level: float,
    ) -> list[dict[str, Any]]:
    """Create dict.

    Parameters:
    -----------
    percentiles : list[int | float]:
        List of percentiles being analyzed

    percentiles_estimates: list[NDArray[np.float64]]:
        list of all percentiles estimates

    percentile_method : str
        Method used for percentile calculation

    ci_level : float
        Confidence interval level

    Returns:
    --------
    list[dict]: list of dictionaries. Each dictionary contains statistics for a specific percentile
    """

    # Initialize dictionary
    dict_with_percentiles_statistics: list[dict[str, Any]] = []

    # Convert list into numpy array for better indexing
    percentiles_estimates_stack: NDArray[np.floating[Any]] =  np.vstack(percentiles_estimates)

    # Iterate over percentiles and compute statistics
    for i, p in enumerate(percentiles):

        # Get estimates for current percentile
        percentile_estimates = percentiles_estimates_stack[:, i]

        # Compute and store statistics
        dict_with_percentiles_statistics.append(_compute_percentile_statistics(
            percentile=p,
            percentile_estimates=percentile_estimates,
            percentile_method=percentile_method,
            ci_level=ci_level,
        ))

    return dict_with_percentiles_statistics


def compute_bootstrap_percentiles(
    data_dict: dict[str, Any],
) -> tuple[dict[str, Any], list[NDArray[np.integer[Any] | np.floating[Any]]]]:
    """
    Compute bootstrap percentiles and confidence intervals.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing data

    Returns:
    --------
    dict : Dict with percentiles and confidence intervals
    dict : All bootstrap samples for further analysis
    """
    # Extract data from dictionary
    metric_config: dict[str, Any] = data_dict.get("metric_config", {})
    clean: dict[str, Any] = data_dict.get("clean", {})
    data: NDArray[np.integer[Any] | np.floating[Any]] = clean.get("data", np.array([]))
    requested_percentiles: list[int | float] = metric_config.get("requested_percentiles", [5, 25, 50, 75, 95])
    n_replicates: int = metric_config.get("bootstrap_n_replicates", 10000)
    n_replicate_size: int = metric_config.get("bootstrap_n_replicate_size", data.size)
    ci_level: float = metric_config.get("bootstrap_ci_level", 0.95)
    metric_type: str | None = metric_config.get("metric_type")
    metric_precision: int = metric_config.get("metric_precision", 2)
    random_state: int = metric_config.get("random_state", 42)

    # Raise error if something crucial is missing
    if any(map(is_falsy, ( metric_config, clean, data ))):
        raise ValueError("---> The data dictionary does not contain all required parts.")

    # Initialize random generator
    rng = np.random.default_rng(random_state)

    # Define percentile method based on metric_precision
    percentile_method: str = "linear" if metric_type == "continuous" else "nearest"

    # Generate all percentiles from 0 to 100 in steps of 5
    all_percentiles: list[int | float] = list(range(0, 101, 5))

    # Initialize lists
    bootstrap_samples: list[NDArray[np.integer[Any] | np.floating[Any]]] = []
    bootstrap_sample: NDArray[np.integer[Any] | np.floating[Any]] = np.array([])
    computed_all_percentiles: list[NDArray[np.float64]] = []
    computed_requested_percentiles: list[NDArray[np.float64]] = []

    # Bootstrap resampling
    for _ in range(n_replicates):

        # Generate bootstrap sample
        bootstrap_sample = rng.choice(data, size=n_replicate_size, replace=True)

        # Store bootstrap sample for further analysis if needed
        bootstrap_samples.append(bootstrap_sample)

        # Compute all percentiles for current bootstrap sample and add to list
        computed_all_percentiles\
            .append(np.percentile(bootstrap_sample, all_percentiles, method=percentile_method))

        # Compute requested percentiles for current bootstrap sample and add to list
        computed_requested_percentiles\
            .append(np.percentile(bootstrap_sample, requested_percentiles, method=percentile_method))

    # Create list for all bootstrap percentile statistics
    all_bootstrap_percentiles: list[dict[str, Any]] = _create_percentile_statistics_list(
        percentiles=all_percentiles,
        percentiles_estimates=computed_all_percentiles,
        percentile_method=percentile_method,
        ci_level=ci_level,
    )

    # Create list for requested bootstrap percentile statistics
    requested_bootstrap_percentiles: list[dict[str, Any]] = _create_percentile_statistics_list(
        percentiles=requested_percentiles,
        percentiles_estimates=computed_requested_percentiles,
        percentile_method=percentile_method,
        ci_level=ci_level,
    )

    # Compute cutoffs for normative tables based on requested percentiles
    percentile_cutoffs = _compute_cutoffs(requested_bootstrap_percentiles, metric_precision=metric_precision)

    # Store results in data dictionary
    data_dict["bootstrap"] = {
        "all_percentiles": all_bootstrap_percentiles,
        "requested_percentiles": requested_bootstrap_percentiles,
        "cutoffs": percentile_cutoffs,
    }

    return data_dict, bootstrap_samples
