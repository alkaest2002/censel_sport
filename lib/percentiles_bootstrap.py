
from typing import Any, cast

import numpy as np


def compute_bootstrap_percentiles(
    data: np.ndarray,
    requested_percentiles: list[int] | None = None,
    n_replicates: int = 10000,
    n_replicate_size: int | None = None,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> tuple[dict[str, Any],dict[str, list[float]]]:
    """
    Compute percentile-based normative table using bootstrap with confidence intervals.

    Parameters:
    -----------
    data : array-like
        Observed running times or count data
    requested_percentiles : list
        List of requested percentiles to compute (e.g., [5, 10, 25, 50, 75, 90, 95])
    n_replicates : int
        Number of bootstrap replicates
    n_replicate_size : int
        Size of each bootstrap replicate (default 100)
    ci_level : float
        Confidence level for intervals (default 0.95 for 95% CI)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    dict : Normative table with percentiles and confidence intervals
    dict : All bootstrap replicates for further analysis
    """
    if requested_percentiles is None:
        # 0-5, 5-25, 25-50, 50-75, 75-95, 95-100
        requested_percentiles = [5, 25, 50, 75, 95]

    if n_replicate_size is not None:
        n_replicate_size = len(data)

    # Initialize random generator
    rng = np.random.default_rng(random_state)

    # Inittialize variables
    results: list[dict[str, str | float]] = []
    bootstrap_estimates: dict[str, list[float]] = {f"p{p}": [] for p in requested_percentiles}
    alpha = 1 - ci_level
    lower_ci = (alpha / 2) * 100
    upper_ci = (1 - alpha / 2) * 100

    # Bootstrap resampling
    for _ in range(n_replicates):
        # Generate bootstrap sample with replacement
        resample = rng.choice(data, size=n_replicate_size, replace=True, shuffle=True)

        # Compute percentiles for this bootstrap sample
        for p in requested_percentiles:
            computed_percentile = cast("float", np.percentile(resample, p))
            bootstrap_estimates[f"p{p}"].append(computed_percentile)

    for p in requested_percentiles:
        # Convert to numpy array for easier calculations
        estimates = np.array(bootstrap_estimates[f"p{p}"])

        # Append results
        results.append({
            "percentile": p,
            "value": np.median(estimates),
            "ci_level": ci_level,
            "ci_lower": np.percentile(estimates, lower_ci),
            "ci_upper": np.percentile(estimates, upper_ci),
            "std_error": np.std(estimates),
        })

    return {
        "computed_percentiles": results,
        "n_replicates": n_replicates,
    }, bootstrap_estimates
