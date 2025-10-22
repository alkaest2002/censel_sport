
from itertools import pairwise
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


def _compute_percentile_cutoffs(
        bootstrap_percentiles: list[dict[str, Any]],
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
    percentiles_values: list[int | float] = [percentile["value"] for percentile in bootstrap_percentiles]

    # Update percentile values with lowers and upper bounds, rounded to specified precision
    corrected_percentiles_values = np.round([0, *percentiles_values, 1e10], metric_precision)

    # Compute cutoffs in the form of: [(lower_bound, upper_bound), ...]
    return list(pairwise(corrected_percentiles_values))

def compute_bootstrap_percentiles(
    data_dict: dict[str, Any],
) -> tuple[dict[str, Any], list[NDArray[np.integer[Any] | np.floating[Any]]]]:
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
    # Extract from data dictionary
    clean: dict[str, Any] = data_dict.get("clean", {})
    data: NDArray[np.integer[Any] | np.floating[Any]] = clean.get("data", np.array([]))
    metric_config: dict[str, Any] = data_dict.get("metric_config", {})
    requested_percentiles: list[int | float] = metric_config.get("requested_percentiles", [5, 25, 50, 75, 95])
    n_replicates: int = metric_config.get("bootstrap_n_replicates", 10000)
    n_replicate_size: int = metric_config.get("bootstrap_n_replicate_size", data.size)
    ci_level: float = metric_config.get("bootstrap_ci_level", 0.95)
    metric_precision: int = metric_config.get("metric_precision", 2)
    random_state: int = metric_config.get("random_state", 42)

    # Initialize random generator
    rng = np.random.default_rng(random_state)


    # If data is empty, raise error
    if data.size == 0:
        raise ValueError("No valid data available for bootstrap sampling.")

    # Inittialize variables
    boostrap_samples: list[NDArray[np.integer[Any] | np.floating[Any]]] = []
    bootstrap_estimates: dict[str, list[float]] = {f"p{p}": [] for p in requested_percentiles}
    bootstrap_percentiles: list[dict[str, Any]] = []
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
    data_dict["bootstrap"] = {
        "percentiles": bootstrap_percentiles,
        "cutoffs": percentile_cutoffs,
    }

    return data_dict, boostrap_samples
