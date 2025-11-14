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
) -> tuple[dict[str, Any], list[NDArray[np.number[Any]]]]:
    """Validate bootstrap percentiles using Monte Carlo simulation from fitted distribution.

    This function performs Monte Carlo validation by generating synthetic datasets
    from the best-fitted distribution and computing validation statistics to assess
    the quality of bootstrap percentile estimates.

    Args:
        data_dict: Dictionary containing analysis data with the following required keys:
            - metric_config: Configuration parameters including metric_type,
              montecarlo_n_samples, montecarlo_n_size, and random_state
            - clean: Cleaned data containing the original dataset
            - bootstrap: Bootstrap results with requested percentiles
            - fit: Distribution fitting results with best model information

    Returns:
        A tuple containing:
            - Updated data dictionary with Monte Carlo validation results
            - List of synthetic datasets generated during validation

    Raises:
        ValueError: If any required data dictionary components are missing or empty.

    Note:
        The function uses different percentile computation methods based on metric type:
        'linear' for continuous metrics and 'nearest' for discrete metrics.
    """
    # Extract data from dictionary
    metric_config: dict[str, Any] = data_dict.get("metric_config", {})
    clean: dict[str, Any] = data_dict.get("clean", {})
    bootstrap: dict[str, Any] = data_dict.get("bootstrap", {})
    fitted_distributions: dict[str, Any] = data_dict.get("fit", {})
    metric_type: Literal["continuous", "discrete"] | None = metric_config.get("metric_type")
    montecarlo_n_samples: int = metric_config.get("montecarlo_n_samples", 0)
    montecarlo_n_size: int = metric_config.get("montecarlo_n_size", 0)
    random_state: int = metric_config.get("random_state", 42)
    data: NDArray[np.number[Any]] = clean.get("data", [])
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
            data,
            bootstrap_requested_percentiles,
            best_model,
        ),
    )):
        raise ValueError("---> The data dictionary does not contain all required parts.")

    # Get distributions
    distributions: dict[str, stats.rv_continuous | stats.rv_discrete] =\
        get_continuous_distributions() if metric_type == "continuous" else get_discrete_distributions()

    # Get best model distribution class
    model_class: type[stats.rv_continuous | stats.rv_discrete] = distributions[best_model["name"]]

    # Instantiate best model class with fitted parameters
    model: stats.rv_continuous | stats.rv_discrete = model_class(*best_model["parameters"])

    # Initialize list to store montecarlo samples
    montecarlo_samples: list[NDArray[np.number[Any]]] = []

    # Initialize list to store montecarlo percentile estimates
    montecarlo_estimates: dict[int | float, list[float]] = {
        percentile_data["percentile"]: []
            for percentile_data in bootstrap_requested_percentiles
    }

    # Initialize list to store montecarlo percentile estimates metrics
    montecarlo_results: list[dict[str, Any]] = []

    # Define percentile method based on metric_type
    percentile_method: Literal["linear", "nearest"] = "linear" if metric_type == "continuous" else "nearest"

    # Run Monte Carlo simulations
    for i in range(montecarlo_n_samples):

        # Generate synthetic dataset from fitted distribution
        synthetic_data: NDArray[np.number[Any]] = model.rvs(size=montecarlo_n_size, random_state=random_state + i)

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
        montecarlo_values: NDArray[np.number[Any]] =\
            np.array(montecarlo_estimates[percentile])

        # Compute montecarlo values
        montecarlo_value: float = np.percentile(montecarlo_values, 50, method=percentile_method)
        montecarlo_min: float = np.min(montecarlo_values)
        montecarlo_max: float = np.max(montecarlo_values)
        montecarlo_first_quartile: float = np.percentile(montecarlo_values, 25, method=percentile_method)
        montecarlo_third_quartile: float = np.percentile(montecarlo_values, 75, method=percentile_method)
        montecarlo_iqr: float = montecarlo_third_quartile - montecarlo_first_quartile
        bias: float = montecarlo_value - bootstrap_value
        relative_bias: float = (bias / bootstrap_value) * 100 if bootstrap_value != 0 else 0
        coverage: float = (np.mean((montecarlo_values >= bootstrap_ci_lower) &
            (montecarlo_values <= bootstrap_ci_upper)) * 100).astype(float)

        montecarlo_results.append({
            "percentile": percentile,
            "value": montecarlo_value,
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
