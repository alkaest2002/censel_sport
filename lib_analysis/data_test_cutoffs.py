from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from lib_analysis.utils_generic import is_falsy
from lib_analysis.utils_stats import apply_standardization


def _apply_cutoffs(
    requested_percentiles: list[float],
    cutoffs: list[tuple],
    data: NDArray[np.number[Any]],
    awarded_scores: list[float],
    higher_is_better: bool,
    sample_sizes: list[int],
    random_state: int,
) -> pd.DataFrame:
    """Apply cutoffs to several random samples of different sizes and collect results.

    Args:
        requested_percentiles: List of percentile thresholds for cutoffs.
        cutoffs: List of tuples defining the cutoff boundaries.
        data: Array of numerical data to sample from.
        awarded_scores: List of scores to assign for each cutoff range.
        higher_is_better: Whether higher values indicate better performance.
        sample_sizes: List of sample sizes to test.
        random_state: Seed for random number generation.

    Returns:
        List of dictionaries containing statistical results for each sample size
        and percentile step combination, including p05, p50, p95 values.
    """

    # Initialize random generator
    rng: np.random.Generator = np.random.default_rng(random_state)

    # Create expected percentile proportions
    expected_percentile_proportions: list[float] =\
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

        # Iterate over 1000 samples of current size
        for _ in range(1000):

            # randomly sample n data points
            sampled_data: NDArray[np.number[Any]] = rng.choice(data, size=n, replace=True)

            # Compute standardized scores
            collected_current_sample_size_data.append(
                apply_standardization(
                    data_to_standardize=sampled_data,
                    cutoffs=cutoffs,
                    awarded_scores=awarded_scores,
                    higher_is_better=higher_is_better,
                )
                .loc[:, "standardized_step"]
                    .value_counts(normalize=True)
                    .reindex(range(1, len(cutoffs)+1), fill_value=0) # Make sure every standardize step is in
                    .rename("observed_proportion")
                    .to_frame()
                    .sort_index()
                    .reset_index(drop=False, names="perc_step"))

        # Convert collected data to DataFrame
        current_sample_size_df: pd.DataFrame = (
            pd.concat(collected_current_sample_size_data)
                .assign(sample_size=n)
                .loc[:, ["sample_size", "perc_step", "observed_proportion"]]
            )

        # append current sample size data to collected data
        collected_data.append(current_sample_size_df)

    # Collect results
    results: pd.DataFrame = (
        pd.concat(collected_data)
            .groupby(["sample_size", "perc_step"], as_index=False)
            .agg({
                "observed_proportion": [
                    ("p05", lambda x: round(x.quantile(0.05)*100,2)), # type: ignore[list-item]
                    ("p95", lambda x: round(x.quantile(0.95)*100,2)), # type: ignore[list-item]
                    ("p50", lambda x: round(x.quantile(0.50)*100,2)), # type: ignore[list-item]
                ],
            })
            .assign(expected_proportion=
                lambda df: df["perc_step"].apply(lambda x: expected_percentile_proportions[int(x) - 1]))
        )

    # Better columns names
    results.columns=[ s if s else f for f,s in results.columns.to_flat_index() ]


    return results.loc[:, ["sample_size", "perc_step", "expected_proportion", "p50", "p05", "p95"]]


def bootstrap_test_cutoffs(
    data_dict: dict[str, Any],
) -> dict[str, Any]:
    """Bootstrap test cutoffs using different sample sizes.

    Args:
        data_dict: Dictionary containing data and configuration parameters.
                  Must include 'metric_config', 'clean', and 'bootstrap' sections.
                  Expected structure:
                  - metric_config: Contains higher_is_better, requested_percentiles, random_state
                  - clean: Contains the data array
                  - bootstrap: Contains cutoffs configuration

    Returns:
        Updated data dictionary with bootstrap test results added under
        data_dict["bootstrap"]["cutoffs_test"]. The results include:
        - sample_sizes: List of tested sample sizes
        - percentile_bands: List of percentile band tuples
        - results: Statistical results from the bootstrap analysis

    Raises:
        ValueError: If required data components (data, cutoffs, requested_percentiles)
                   are missing or empty.

    Note:
        The function tests sample sizes of [30, 50, 100, 150, 200, 300] by default
        and performs 1000 bootstrap iterations for each sample size.
    """
    # Extract data from dictionary
    metric_config: dict[str, Any] =  data_dict.get("metric_config", {})
    clean: dict[str, Any] = data_dict.get("clean", {})
    data: NDArray[np.number[Any]] = clean.get("data", np.array([]))
    bootstrap: dict[str, Any] = data_dict.get("bootstrap", {})
    cutoffs = bootstrap.get("cutoffs", {})
    awarded_scores: list[float] = metric_config.get("awarded_scores", [])
    higher_is_better: bool = metric_config.get("higher_is_better", False)
    requested_percentiles: list[float] = sorted(metric_config.get("requested_percentiles", []))
    random_state: int = metric_config.get("random_state", 42)

    # Raise error if something crucial is missing
    if any(map(is_falsy, (metric_config, clean, data, bootstrap, cutoffs, requested_percentiles))):
        raise ValueError("---> The data dictionary does not contain all required parts.")

    # Define base sample_sizes
    sample_sizes: list[int] = [30, 50, 100, 150, 200, 300, 500]

    # Define percentile bands
    percentile_bands: list[tuple[float, float]] = list(pairwise([0, *requested_percentiles, 100]))

    # Apply cutoffs to data
    final_data: pd.DataFrame = _apply_cutoffs(
        requested_percentiles=requested_percentiles,
        cutoffs=cutoffs,
        data=data,
        awarded_scores=awarded_scores,
        higher_is_better=higher_is_better,
        sample_sizes=sample_sizes,
        random_state=random_state,
    )

    # Update data dictionary
    data_dict["bootstrap"]["cutoffs_test"] = {
        "sample_sizes": sample_sizes,
        "percentile_bands": percentile_bands,
        "results": final_data,
    }

    return data_dict
