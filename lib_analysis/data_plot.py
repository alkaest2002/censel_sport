


from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from lib_analysis.utils_distributions import DistributionType, FitFunctionType, get_distributions
from lib_analysis.utils_generic import is_falsy
from lib_analysis.utils_plots import (
    plot_bootstrap_percentile_with_ci,
    plot_hanging_rootogram,
    plot_histogram_with_fitted_model,
    plot_montecarlo,
    plot_qq_plot,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

def create_plots(data_dict: dict[str, Any]) -> dict[str, Any]:

    # Extract data from dictionary
    metric_config: dict[str, Any] = data_dict.get("metric_config", {})
    clean: dict[str, Any] = data_dict.get("clean", {})
    bootstrap: dict[str, Any] = data_dict.get("bootstrap", {})
    montecarlo: dict[str, Any] = data_dict.get("montecarlo", {})
    fit: dict[str, Any] = data_dict.get("fit", {})
    metric_type: Literal["time", "count"] | None = metric_config.get("metric_type")
    data: NDArray[np.integer[Any] | np.floating[Any]] = clean.get("data", np.array([]))
    bootstrap_percentiles: list[dict[str, Any]] = bootstrap.get("percentiles", [])
    best_model: dict[str, Any] = fit.get("best_model", {})
    best_model_name: str = best_model.get("name", "")
    best_model_parameters: list[float] = best_model.get("parameters", [])
    montecarlo_results: list[dict[str, Any]] = montecarlo.get("results", [])

    # Raise error if something is missing
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
                    bootstrap_percentiles,
                    best_model,
                    best_model_name,
                    best_model_parameters,
                    montecarlo_results,
                ),
            ),
        ):
        raise ValueError("---> The data dictionary does not contain all required parts.")

    # Get distributions
    distributions: dict[str, tuple[DistributionType, FitFunctionType]] =\
        get_distributions(metric_type, best_model_name)

    # Get best model class
    model_class, _ = distributions[best_model_name]

    # Instantiate best model class with fitted params
    try:
        model = model_class(*best_model_parameters)
    except (TypeError, ValueError) as e:
        raise ValueError(f"---> Failed to instantiate model {best_model_name}: {e}") from e

    # Add plots
    data_dict["plots"] = [
        {
            "name": "histogram_of_data_with_fitted_distribution",
            "svg": plot_histogram_with_fitted_model(data, best_model_name, model),
        },
        {
            "name": "histogram_of_percentiles_with_ci",
            "svg": plot_bootstrap_percentile_with_ci(bootstrap_percentiles),
        },
        {
            "name": "monte_carlo_vs_bootstrap_percentiles",
            "svg": plot_montecarlo(montecarlo_results),
        },
    ]

    if metric_type == "time":
        data_dict["plots"].append({
            "name": "qq_plot",
            "svg": plot_qq_plot(data, best_model_name, model),
        })

    if metric_type == "count":
         data_dict["plots"].append({
            "name": "rootogram",
            "svg": plot_hanging_rootogram(data, best_model_name, model),
        })

    return data_dict
