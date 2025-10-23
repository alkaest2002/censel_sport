from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from lib.utils_distributions import DistributionType, FitFunctionType, get_distributions
from lib.utils_generic import is_falsy


def monte_carlo_validation(
    data_dict: dict[str, Any],
) -> tuple[dict[str, Any], list[NDArray[np.integer[Any] | np.floating[Any]]]]:
    """
    Validate bootstrap percentiles using Monte Carlo simulation from fitted distribution.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing data

    Returns:
    --------
    dict : Updated data dictionary
    """
    # Extract data from dictionary
    metric_config: dict[str, Any] = data_dict.get("metric_config", {})
    clean: dict[str, Any] = data_dict.get("clean", {})
    bootstrap: dict[str, Any] = data_dict.get("bootstrap", {})
    fitted_distributions: dict[str, Any] = data_dict.get("fit", {})
    metric_type: Literal["time", "count"] | None = metric_config.get("metric_type")
    requested_percentiles: list[int | float] = metric_config.get("requested_percentiles", [])
    montecarlo_n_samples: int = metric_config.get("montecarlo_n_samples", 0)
    montecarlo_n_size: int = metric_config.get("montecarlo_n_size", 0)
    random_state: int = metric_config.get("random_state", 0)
    data: NDArray[np.integer[Any] | np.floating[Any]] = clean.get("data", [])
    bootstrap_percentiles: dict[str, Any] = bootstrap.get("percentiles", {})
    best_model: dict[str, Any] = fitted_distributions.get("best_model", {})

    # Raise error if something is missing
    if any(map(is_falsy,
        (
            metric_config,
            clean,
            bootstrap,
            fitted_distributions,
            metric_type,
            requested_percentiles,
            montecarlo_n_samples,
            montecarlo_n_size,
            random_state,
            data,
            bootstrap_percentiles,
            best_model,
        ),
    )):
        raise ValueError("---> The data dictionary does not contain all required parts.")

    # Get distributions
    distributions: dict[str, tuple[DistributionType, FitFunctionType]] =\
        get_distributions(metric_type, best_model["name"])

    # Get best model class
    model_class, _ = distributions[best_model["name"]]

    # Istantiate best model class with fitted parameters
    model = model_class(*best_model["parameters"])

    # Init lists to store montecarlo samples and synthetic percentile estimates
    montecarlo_samples: list[NDArray[np.integer[Any] | np.floating[Any]]] = []
    montecarlo_estimates: dict[int | float, list[float]] = {p: [] for p in requested_percentiles}

    # Compute validation metrics
    validation_results: list[dict[str, Any]] = []

    # Run Monte Carlo simulations
    for i in range(montecarlo_n_samples):

        # Generate synthetic dataset from fitted distribution
        synthetic_data = model.rvs(size=montecarlo_n_size, random_state=random_state + i)

        # Append sample to list
        montecarlo_samples.append(synthetic_data)

        # Compute percentiles
        for p in requested_percentiles:
            montecarlo_estimates[p].append(np.percentile(synthetic_data, p))

    # For each percentile, compute bias, RMSE, coverage
    for p in requested_percentiles:

        # Get bootstrap percentile info
        bootstrap_percentile: dict[str, Any] =\
            next((x for x in bootstrap_percentiles if x["percentile"] == p), None) # type: ignore[arg-type, comparison-overlap, index, misc]
        bootstrap_value: float = bootstrap_percentile["value"]
        bootstrap_ci_lower: float = bootstrap_percentile["ci_lower"]
        bootstrap_ci_upper: float = bootstrap_percentile["ci_upper"]

        # Get synthetic values
        montecarlo_values: NDArray[np.integer[Any] | np.floating[Any]] = np.array(montecarlo_estimates[p])

        # Bias
        bias = np.mean(montecarlo_values) - bootstrap_value

        # Relative bias (as percentage)
        relative_bias = (bias / bootstrap_value) * 100 if bootstrap_value != 0 else 0

        # RMSE
        rmse = np.sqrt(np.mean((montecarlo_values - bootstrap_value)**2))

        # Coverage: % of synthetic values within bootstrap CI
        coverage = np.mean((montecarlo_values >= bootstrap_ci_lower) &
            (montecarlo_values <= bootstrap_ci_upper)) * 100

        validation_results.append({
            "percentile": f"{p}",
            "bootstrap_value": bootstrap_value,
            "bootstrap_ci_lower": bootstrap_ci_lower,
            "bootstrap_ci_upper": bootstrap_ci_upper,
            "montecarlo_value": np.mean(montecarlo_values),
            "montecarlo_std": np.std(montecarlo_values),
            "montecarlo_min": np.min(montecarlo_values),
            "montecarlo_max": np.max(montecarlo_values),
            "montecarlo_iqr": np.percentile(montecarlo_values, 75) - np.percentile(montecarlo_values, 25),
            "bias": bias,
            "relative_bias_%": relative_bias,
            "rmse": rmse,
            "coverage_%": coverage,
        })

    # Update data dictionary
    data_dict["montecarlo"] = {
        "results": validation_results,
    }

    return data_dict, montecarlo_samples
