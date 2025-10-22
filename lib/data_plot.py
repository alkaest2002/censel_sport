


from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from lib.utils_distributions import DistributionType, FitFunctionType, get_distributions
from lib.utils_generic import is_falsy
from lib.utils_plots import plot_qq_plot

if TYPE_CHECKING:
    from numpy.typing import NDArray

def create_plots(data_dict: dict[str, Any]) -> dict[str, Any]:

    # Extract from dictionary
    metric_config: dict[str, Any] = data_dict.get("metric_config", {})
    metric_type: Literal["time", "count"] | None = metric_config.get("metric_type")
    clean: dict[str, Any] = data_dict.get("clean", {})
    fit: dict[str, Any] = data_dict.get("fit", {})
    data: NDArray[np.integer[Any] | np.floating[Any]] = clean.get("data", np.array([]))
    best_model: dict[str, Any] = fit.get("best_model", {})
    best_model_name: str = best_model.get("name", "")
    best_model_parameters: list[float] = best_model.get("parameters", [])

    # Raise error if something is missing
    if any(map(is_falsy,
               (
                   metric_config,
                   clean,
                   data,
                   metric_type,
                   fit,
                   data,
                   best_model,
                   best_model_name,
                   best_model_parameters,
                ),
            ),
        ):
        raise ValueError("The data dictionary does not contain all required parts.")

    # Get distributions
    distributions: dict[str, tuple[DistributionType, FitFunctionType]] =\
        get_distributions(metric_type, best_model_name)

    # Get best model class
    model_class, _ = distributions[best_model_name]

    # Instantiate best model class with fitted params
    try:
        model = model_class(*best_model_parameters)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Failed to instantiate model {best_model_name}: {e}") from e

    # Add plots
    if metric_type == "time":
        data_dict["plots"] = {
            "qq_plot": plot_qq_plot(data, best_model_name, model),
        }

    return data_dict
