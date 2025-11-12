from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from lib_analysis.utils_generic import is_falsy
from lib_analysis.utils_stats import apply_standardization


def _apply_cutoffs(
    metric_title: str,
    higher_is_better: bool,
    requested_percentiles: list[float],
    cutoffs: list[tuple],
    data: NDArray[np.integer[Any] | np.floating[Any]],
    random_state: int,
) -> list:
    """_apply_cutoffs to several random samples of different sizes and collect results."""

    # Initialize random generator
    rng = np.random.default_rng(random_state)

    # Define base sample_sizes
    sample_sizes = [30, 50, 100, 150, 300]

    # Omit sample sizes larger than data length
    sample_sizes = list(filter(lambda x: x <= len(data), sample_sizes))

    # Create expected percentiles: i.e., [5, 20, 25, 25, 20, 5]
    expected_percentile: list[float] =\
        np.subtract(
            np.array([*requested_percentiles, 100]),
            np.array([0, *requested_percentiles]),
        ).tolist()

    # Initialize collected data
    collected_data: list[pd.DataFrame] = []

    # Iterate over sample sizes
    for n in sample_sizes:

        # Initialize collected data for current sample size
        collected_current_sample_size_data: list[pd.DataFrame] = []

        # Iterate over 500 samples of current size
        for _ in range(500):

            # randomly sample n data points
            sampled_data = rng.choice(data, size=n, replace=True).tolist()

            # Compute standardized scores
            collected_current_sample_size_data.append(
                apply_standardization(
                    data_to_standardize=np.array(sampled_data),
                    cutoffs=cutoffs,
                    higher_is_better=higher_is_better,
                )
                .drop(["standardized_value_bounds"], axis=1)
                .loc[:, "standardized_value"]
                    .value_counts(normalize=True)
                .rename("value")
                .to_frame()
                .sort_index()
                .reset_index(drop=False, names="perc_step"))

        # Convert collected data to DataFrame
        current_sample_size_df: pd.DataFrame = (
            pd.concat(collected_current_sample_size_data)
                .assign(
                    sample_size=n,
                    test=metric_title,
                )
                .loc[:, ["test", "sample_size", "perc_step", "value"]]
            )

        # append current sample size data to collected data
        collected_data.append(current_sample_size_df)

    # Collect results
    results: pd.DataFrame = (
        pd.concat(collected_data)
            .groupby(["test", "sample_size", "perc_step"], as_index=False)
            .agg({
                "value": [
                    ("p5", lambda x: round(x.quantile(0.05)*100,2)),  # type: ignore[list-item]
                    ("p95", lambda x: round(x.quantile(0.95)*100,2)), # type: ignore[list-item]
                    ("p50", lambda x: round(x.quantile(0.50)*100,2)), # type: ignore[list-item]
                ],
            })
            .assign(perc_expected=
                lambda df: df["perc_step"].apply(lambda x: expected_percentile[int(x) - 1]))
        )

    # Better columns names
    results.columns=[ s if s else f for f,s in results.columns.to_flat_index() ]


    return (results\
        .loc[:, ["test", "sample_size", "perc_step", "perc_expected", "p50", "p5", "p95"]]
        .to_dict(orient="records")
    )


def bootstrap_test_cutoffs(
    data_dict: dict[str, Any],
) -> dict[str, Any]:
    """
    Clean performance data by removing outliers and invalid values.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing data

    Returns:
    --------
    dict : Updated data dictionary
    """
    # Extract data from dictionary
    metric_config: dict[str, Any] =  data_dict.get("metric_config", {})
    clean: dict[str, Any] = data_dict.get("clean", {})
    bootstrap: dict[str, Any] = data_dict.get("bootstrap", {})
    data: NDArray[np.integer[Any] | np.floating[Any]] = clean.get("data", np.array([]))
    cutoffs = bootstrap.get("cutoffs", {})
    metric_title: str = metric_config.get("title", "")
    higher_is_better: bool = metric_config.get("higher_is_better", False)
    requested_percentiles: list[float] = metric_config.get("requested_percentiles", [])
    random_state: int = metric_config.get("random_state", 42)

    # Raise error if something is missing
    if any(map(is_falsy, (metric_title, data, cutoffs, requested_percentiles))):
        raise ValueError("---> The data dictionary does not contain all required parts.")

    # Apply cutoffs to data
    final_data: list[dict[str, Any]] = _apply_cutoffs(
        metric_title=metric_title,
        higher_is_better=higher_is_better,
        requested_percentiles=requested_percentiles,
        cutoffs=cutoffs,
        data=data,
        random_state=random_state,
    )

    # Update data dictionary
    data_dict["bootstrap"]["cutoffs_test"] = final_data

    return data_dict
