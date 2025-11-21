from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from lib_analysis.utils_distributions_continuous import get_continuous_distributions
from lib_analysis.utils_distributions_discrete import get_discrete_distributions
from lib_analysis.utils_generic import is_falsy
from lib_analysis.utils_plots import (
    plot_bootstrap_percentile_with_ci,
    plot_hanging_rootogram,
    plot_histogram_with_fitted_model,
    plot_montecarlo_vs_bootstrap,
    plot_qq_plot,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy import stats


def create_plots(data_dict: dict[str, Any]) -> dict[str, Any]:
    """Create plots for the analysis results.

    Args:
        data_dict: Dictionary containing data.

    Returns:
        Updated data dictionary with added 'plots' key containing a list of plot
        dictionaries, each with 'name' and 'svg' keys.

    Raises:
        ValueError: If the data dictionary is missing required components or if
            model instantiation fails.
    """
    # Extract data from dictionary
    metric_config: dict[str, Any] = data_dict.get("metric_config", {})
    clean: dict[str, Any] = data_dict.get("clean", {})
    bootstrap: dict[str, Any] = data_dict.get("bootstrap", {})
    montecarlo: dict[str, Any] = data_dict.get("montecarlo", {})
    fit: dict[str, Any] = data_dict.get("fit", {})
    metric_type: Literal["continuous", "discrete"] | None = metric_config.get("metric_type")
    data: NDArray[np.number[Any]] = clean.get("data", np.array([]))
    bootstrap_requested_percentiles: list[dict[str, Any]] = bootstrap.get("requested_percentiles", [])
    bootstrap_all_percentiles: list[dict[str, Any]] = bootstrap.get("all_percentiles", [])
    best_model: dict[str, Any] = fit.get("best_model", {})
    best_model_name: str = best_model.get("name", "")
    best_model_parameters: list[float] = best_model.get("parameters", [])
    montecarlo_percentiles: list[dict[str, Any]] = montecarlo.get("results", [])

    # Raise error if something crucial is missing
    if any(map(is_falsy,
            (
                metric_config,
                clean,
                bootstrap,
                montecarlo,
                fit,
                data,
                metric_type,
                data,
                bootstrap_requested_percentiles,
                bootstrap_all_percentiles,
                best_model,
                best_model_name,
                best_model_parameters,
                montecarlo_percentiles,
            ),
        )):
        raise ValueError("---> The data dictionary does not contain all required parts.")

    # Get distributions
    distributions: dict[str, stats.rv_continuous | stats.rv_discrete] = (
        get_continuous_distributions() if metric_type == "continuous"
            else get_discrete_distributions()
    )

    # Get best model class
    model_class: type[stats.rv_continuous | stats.rv_discrete] = distributions[best_model_name]

    try:
        # Instantiate best model class with fitted params
        model: stats.rv_continuous | stats.rv_discrete = model_class(*best_model_parameters)

    # Handle potential errors during instantiation
    except (TypeError, ValueError) as e:
        raise ValueError(f"Failed to instantiate model {best_model_name}: {e}") from e

    # Add common plots for both continuous and discrete distrubutions
    data_dict["plots"] = [
        {
            "name": "histogram_of_data_with_fitted_distribution",
            "svg": plot_histogram_with_fitted_model(data, best_model_name, model),
        },
        {
            "name": "histogram_of_percentiles_with_ci",
            "svg": plot_bootstrap_percentile_with_ci(
                bootstrap_requested_percentiles,
                bootstrap_all_percentiles,
            ),
        },
        {
            "name": "monte_carlo_vs_bootstrap_requested_percentiles",
            "svg": plot_montecarlo_vs_bootstrap(bootstrap_requested_percentiles, montecarlo_percentiles),
        },
    ]

    # Add Q-Q plot for continuous distrubutions
    if metric_type == "continuous":
        data_dict["plots"].append({
            "name": "qq_plot",
            "svg": plot_qq_plot(data, best_model_name, model),
        })

    # Add rootogram for discrete distrubutions
    if metric_type == "discrete":
        data_dict["plots"].append({
            "name": "rootogram",
            "svg": plot_hanging_rootogram(data, best_model_name, model),
        })

    return data_dict
