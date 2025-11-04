# mypy: disable-error-code="call-overload"
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from lib_analysis.utils_distributions_continuous import get_continuous_distributions
from lib_analysis.utils_distributions_discrete import get_discrete_distributions
from lib_analysis.utils_generic import is_falsy

if TYPE_CHECKING:
    from scipy import stats


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
    metric_type: Literal["continuous", "discrete"] | None = metric_config.get("metric_type")
    montecarlo_n_samples: int = metric_config.get("montecarlo_n_samples", 0)
    montecarlo_n_size: int = metric_config.get("montecarlo_n_size", 0)
    random_state: int = metric_config.get("random_state", 0)
    data: NDArray[np.integer[Any] | np.floating[Any]] = clean.get("data", [])
    bootstrap_requested_percentiles: list[dict[str, Any]] = bootstrap.get("requested_percentiles", [])
    best_model: dict[str, Any] = fitted_distributions.get("best_model", {})

    # Raise error if something is missing
    if any(map(is_falsy,
        (
            metric_config,
            clean,
            bootstrap,
            fitted_distributions,
            metric_type,
            montecarlo_n_samples,
            montecarlo_n_size,
            random_state,
            data,
            bootstrap_requested_percentiles,
            best_model,
        ),
    )):
        raise ValueError("---> The data dictionary does not contain all required parts.")

    # Get distributions
    distributions: dict[str, stats.rv_continuous | stats.rv_discrete] =\
        get_continuous_distributions() if metric_type == "continuous" else get_discrete_distributions()

    # Get best model class
    model_class = distributions[best_model["name"]]

    # Istantiate best model class with fitted parameters
    model = model_class(*best_model["parameters"])

    # Init lists to store montecarlo samples and synthetic percentile estimates
    montecarlo_samples: list[NDArray[np.integer[Any] | np.floating[Any]]] = []
    montecarlo_estimates: dict[int | float, list[float]] = {
        percentile_data["percentile"]: []
        for percentile_data in bootstrap_requested_percentiles
    }

    # Compute validation metrics
    montecarlo_results: list[dict[str, Any]] = []

    # Define percentile method based on metric_precision
    percentile_method = "linear" if metric_type == "continuous" else "nearest"

    # Run Monte Carlo simulations
    for i in range(montecarlo_n_samples):

        # Generate synthetic dataset from fitted distribution
        synthetic_data = model.rvs(size=montecarlo_n_size, random_state=random_state + i)

        # Append sample to list
        montecarlo_samples.append(synthetic_data)

        # Compute requested percentiles on synthetic data
        for percentile_data in bootstrap_requested_percentiles:
            p: int | float = percentile_data["percentile"]
            percentile_value: float = np.percentile(synthetic_data, p, method=percentile_method)
            montecarlo_estimates[p].append(percentile_value)

    # Iterate over requested percentiles to compute validation statistics
    for percentile_data in bootstrap_requested_percentiles:

        # Extract bootstrap statistics
        percentile: float = percentile_data["percentile"]
        bootstrap_value: float = percentile_data["value"]
        bootstrap_ci_lower: float = percentile_data["ci_lower"]
        bootstrap_ci_upper: float = percentile_data["ci_upper"]

        # Get synthetic values and convert to numpy array for easier computation
        montecarlo_values: NDArray[np.integer[Any] | np.floating[Any]] =\
            np.array(montecarlo_estimates[percentile])

        # Compute montecarlo values
        montecarlo_value = np.percentile(montecarlo_values, 50, method=percentile_method)
        montecarlo_min = np.min(montecarlo_values)
        montecarlo_max = np.max(montecarlo_values)
        montecarlo_first_quartile = np.percentile(montecarlo_values, 25, method=percentile_method)
        montecarlo_third_quartile = np.percentile(montecarlo_values, 75, method=percentile_method)
        montecarlo_iqr = montecarlo_third_quartile - montecarlo_first_quartile
        bias = montecarlo_value - bootstrap_value
        relative_bias = (bias / bootstrap_value) * 100 if bootstrap_value != 0 else 0
        coverage = np.mean((montecarlo_values >= bootstrap_ci_lower) &
            (montecarlo_values <= bootstrap_ci_upper)) * 100

        montecarlo_results.append({
            "percentile": percentile,
            "value":montecarlo_value,
            "min": montecarlo_min,
            "max": montecarlo_max,
            "first_quartile": montecarlo_first_quartile,
            "third_quartile": montecarlo_third_quartile,
            "iqr": montecarlo_iqr,
            "bias": bias,
            "relative_bias_%": relative_bias,
            "coverage_%": coverage,
        })

    # Update data dictionary
    data_dict["montecarlo"] = {
        "results": montecarlo_results,
    }

    return data_dict, montecarlo_samples
