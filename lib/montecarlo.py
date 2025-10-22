from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from lib.distributions import DistributionType, FitFunctionType, get_distributions
from lib.utils import is_falsy


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
        raise ValueError("The data dictionary does not contain all required parts.")


    # Get distributions
    distributions: dict[str, tuple[DistributionType, FitFunctionType]] =\
        get_distributions(metric_type, best_model["name"])
    model_class, _ = distributions[best_model["name"]]

    # Istantiate best model
    model = model_class(*best_model["params"])

    # Store synthetic percentile estimates
    simulation_samples: list[NDArray[np.integer[Any] | np.floating[Any]]] = []
    simulation_estimates: dict[int | float, list[float]] = {p: [] for p in requested_percentiles}

    # Run Monte Carlo simulations
    for _ in range(montecarlo_n_samples):

        # Generate synthetic dataset from fitted distribution
        synthetic_data = model.rvs(size=montecarlo_n_size, random_state=random_state)

        # Append sample to list
        simulation_samples.append(synthetic_data)

        # Compute percentiles
        for p in requested_percentiles:
            simulation_estimates[p].append(np.percentile(synthetic_data, p))

    # Compute validation metrics
    validation_results: list[dict[str, Any]] = []

    # For each percentile, compute bias, RMSE, coverage
    for p in requested_percentiles:

        # Get bootstrap percentile info
        bootstrap_percentile: dict[str, Any] = bootstrap_percentiles[f"p{p}"]
        bootstrap_value: float = bootstrap_percentile["value"]
        bootstrap_ci_lower: float = bootstrap_percentile["ci_lower"]
        bootstrap_ci_upper: float = bootstrap_percentile["ci_upper"]

        # Get synthetic values
        simulation_values: NDArray[np.integer[Any] | np.floating[Any]] = np.array(simulation_estimates[p])

        # Bias
        bias = np.mean(simulation_values) - bootstrap_value

        # Relative bias (as percentage)
        relative_bias = (bias / bootstrap_value) * 100 if bootstrap_value != 0 else 0

        # RMSE
        rmse = np.sqrt(np.mean((simulation_values - bootstrap_value)**2))

        # Coverage: % of synthetic values within bootstrap CI
        coverage = np.mean((simulation_values >= bootstrap_ci_lower) &
                          (simulation_values <= bootstrap_ci_upper)) * 100

        validation_results.append({
            "percentile": f"{p}",
            "bootstrap_value": bootstrap_value,
            "synthetic_mean": np.mean(simulation_values),
            "bias": bias,
            "relative_bias_%": relative_bias,
            "rmse": rmse,
            "coverage_%": coverage,
            "synthetic_std": np.std(simulation_values),
        })

    # Update data dictionary
    data_dict["simulation"] = {
        "results": validation_results,
    }

    return data_dict, simulation_samples
