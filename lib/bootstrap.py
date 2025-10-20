
from itertools import pairwise
from typing import Any, cast

import numpy as np


def _compute_percentile_cutoffs(
        bootstrap_percentiles: list[dict[str, str | float]],
        metric_precision: int = 2,
    ) -> list[tuple[float, float]]:
    """
    Create a normative table cutoffs using bootstrap percentiles.

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
    percentiles_values = [percentile["value"] for percentile in bootstrap_percentiles]

    # Update percentile values with lowers and upper bounds, rounded to specified precision
    corrected_percentiles_values = np.round([0, *percentiles_values, 1e10], metric_precision)

    # Compute cutoffs in the form of: [(lower_bound, upper_bound), ...]
    return list(pairwise(corrected_percentiles_values))

def compute_bootstrap_percentiles(
    data_dict: dict[str, Any],
) -> tuple[dict[str, Any], list[float]]:
    """
    Compute percentile-based normative table using bootstrap with confidence intervals.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing data

    Returns:
    --------
    dict : Dict with percentiles and confidence intervals
    dict : All bootstrap samples for further analysis
    """
    # Extract parameters from data dictionary
    data = data_dict.get("analysis_data", [])
    requested_percentiles = data_dict.get("metric_config", {}).get("requested_percentiles", [5, 25, 50, 75, 95])
    n_replicates = data_dict.get("metric_config", {}).get("bootstrap_n_replicates", 10000)
    n_replicate_size = data_dict.get("metric_config", {}).get("bootstrap_n_replicate_size", data.size)
    ci_level = data_dict.get("metric_config", {}).get("bootstrap_ci_level", 0.95)
    random_state = data_dict.get("metric_config", {}).get("bootstrap_random_state", 42)
    metric_precision = data_dict.get("metric_config", {}).get("metric_precision", 2)

    # If data is empty, raise error
    if data.size == 0:
        raise ValueError("No valid data available for bootstrap sampling.")

    # Initialize random generator
    rng = np.random.default_rng(random_state)

    # Inittialize variables
    boostrap_samples: list[float] = []
    bootstrap_estimates: dict[str, list[float]] = {f"p{p}": [] for p in requested_percentiles}
    bootstrap_percentiles: list[dict[str, str | float]] = []
    alpha = 1 - ci_level
    lower_ci = (alpha / 2) * 100
    upper_ci = (1 - alpha / 2) * 100

    # Bootstrap resampling
    for _ in range(n_replicates):
        # Generate bootstrap sample with replacement
        resample = rng.choice(data, size=n_replicate_size, replace=True)

        # Store bootstrap sample
        boostrap_samples.append(resample)

        # Compute percentiles for this bootstrap sample
        for p in requested_percentiles:
            computed_percentile = cast("float", np.percentile(resample, p))
            bootstrap_estimates[f"p{p}"].append(computed_percentile)

    for p in requested_percentiles:
        # Convert to numpy array for easier calculations
        estimates = np.array(bootstrap_estimates[f"p{p}"])

        # Append results
        bootstrap_percentiles.append({
            "percentile": p,
            "value": np.median(estimates),
            "ci_level": ci_level,
            "ci_lower": np.percentile(estimates, lower_ci),
            "ci_upper": np.percentile(estimates, upper_ci),
            "std_error": np.std(estimates),
        })

    percentile_cutoffs = _compute_percentile_cutoffs(bootstrap_percentiles, metric_precision=metric_precision)

    # Store results in data dictionary
    data_dict["normative_table"] = {
        "bootstrap_percentiles": bootstrap_percentiles,
        "computed_cutoffs": percentile_cutoffs,
    }

    return data_dict, boostrap_samples
