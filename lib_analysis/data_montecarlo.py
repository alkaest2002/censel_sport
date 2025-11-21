# mypy: disable-error-code="call-overload"
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy import stats

from lib_analysis.utils_distributions_continuous import get_continuous_distributions
from lib_analysis.utils_distributions_discrete import get_discrete_distributions
from lib_analysis.utils_generic import is_falsy
from lib_analysis.utils_stats import compute_sample_size


def monte_carlo_validation(
    data_dict: dict[str, Any],
) -> tuple[dict[str, Any], list[NDArray[np.number[Any]]]]:
    """Validate bootstrap percentiles using Monte Carlo simulation from fitted distribution.

    Args:
        data_dict: Dictionary containing data.

    Returns:
        A tuple containing:
            - Updated data dictionary with Monte Carlo validation results
            - List of synthetic datasets generated during validation

    Raises:
        ValueError: If any required data dictionary components are missing or empty.
    """
    # Extract data from dictionary
    metric_config: dict[str, Any] = data_dict.get("metric_config", {})
    clean: dict[str, Any] = data_dict.get("clean", {})
    bootstrap: dict[str, Any] = data_dict.get("bootstrap", {})
    fitted_distributions: dict[str, Any] = data_dict.get("fit", {})
    metric_type: Literal["continuous", "discrete"] | None = metric_config.get("metric_type")
    data: NDArray[np.number[Any]] = clean.get("data", [])
    montecarlo_n_samples: int = metric_config.get("montecarlo_n_samples", 0)
    random_state: int = metric_config.get("random_state", 42)
    requested_percentiles: list[float] = sorted(metric_config.get("requested_percentiles", []))
    bootstrap_requested_percentiles: pd.DataFrame = bootstrap.get("requested_percentiles", pd.DataFrame())
    best_model: dict[str, Any] = fitted_distributions.get("best_model", {})
    best_model_name: str = best_model.get("name", "")
    best_model_parameters: list[float] = best_model.get("parameters", [])

    # Raise error if something crucial is missing
    if any(map(is_falsy,
        (
            metric_config,
            clean,
            bootstrap,
            data,
            metric_type,
            montecarlo_n_samples,
            requested_percentiles,
            bootstrap_requested_percentiles,
            fitted_distributions,
            best_model,
        ),
    )):
        raise ValueError("---> The data dictionary does not contain all required parts.")

    # Compute sample size based on data
    montecarlo_sample_size: int = compute_sample_size(data_dict)

    # Get distributions
    distributions: dict[str, stats.rv_continuous | stats.rv_discrete] =\
        get_continuous_distributions() if metric_type == "continuous" else get_discrete_distributions()

    # Get best model distribution class
    model_class: type[stats.rv_continuous | stats.rv_discrete] = distributions[best_model_name]

    # Instantiate best model class with fitted parameters
    model: stats.rv_continuous | stats.rv_discrete = model_class(*best_model_parameters)

    # Initialize list to store montecarlo samples
    montecarlo_samples: list[NDArray[np.number[Any]]] = []

    # Initialize list to store montecarlo percentile estimates
    montecarlo_estimates: list[NDArray[np.number[Any]]] = []

    # Define percentile method based on metric_typetype of metric (either 'continuous' or 'discrete')
    percentile_method: Literal["linear", "nearest"] = "linear" if metric_type == "continuous" else "nearest"

    # Run Monte Carlo simulations
    for i in range(montecarlo_n_samples):

        # Generate synthetic dataset from fitted distribution
        synthetic_data: NDArray[np.number[Any]] = model.rvs(size=montecarlo_sample_size, random_state=random_state + i)

        # Append sample to list
        montecarlo_samples.append(synthetic_data)

        # Compute requested percentiles on synthetic data
        montecarlo_estimates.append(
            np.percentile(synthetic_data, requested_percentiles, method=percentile_method))

    # Convert Monte Carlo percentiles estimates to DataFrame for easier aggregation
    montecarlo_df: pd.DataFrame = pd.DataFrame(montecarlo_estimates, columns=requested_percentiles)

    # Compute statistics for each percentile
    montecarlo_stats: pd.DataFrame = (
        pd.DataFrame({
            "percentile": montecarlo_df.columns,
            "value": montecarlo_df.quantile(0.5, interpolation=percentile_method),
            "min": montecarlo_df.min(),
            "max": montecarlo_df.max(),
            "iqr": montecarlo_df.agg(lambda x: stats.iqr(x)),
            "first_quartile": montecarlo_df.quantile(0.25, interpolation=percentile_method),
            "third_quartile": montecarlo_df.quantile(0.75, interpolation=percentile_method),
        })
        .reset_index(drop=True)
        .merge(
            bootstrap_requested_percentiles.loc[:, ["percentile", "value", "ci_lower", "ci_upper"]],
            on="percentile",
            suffixes=("_montecarlo", "_bootstrap"),
        )
        .assign(
            bias=lambda df: df["value_montecarlo"] - df["value_bootstrap"],
            relative_bias=lambda df: df["bias"].div(df["value_bootstrap"]).mul(100).fillna(0),
            coverage=lambda df: [
                montecarlo_df[row["percentile"]].between(
                    row["ci_lower"],
                    row["ci_upper"],
                ).mean() * 100
                for _, row in df.iterrows()
            ],
        )
    )


    # Update metric config with montecarlo_sample_size
    data_dict["metric_config"]["montecarlo_sample_size"] = montecarlo_sample_size

    # Update data dictionary with montecarlo results
    data_dict["montecarlo"] = {
        "results": montecarlo_stats,
    }

    return data_dict, montecarlo_samples
