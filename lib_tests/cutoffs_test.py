from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pandas as pd

from lib_analysis.utils_stats import apply_standardization


def get_cutoffs_test() -> pd.DataFrame:
    """
    Test cutoffs with samples with various sample sizes.
    """

    # Define base sample_sizes
    sample_sizes = [30, 50, 100, 200, 500]

    # Initialize random generator
    rng = np.random.default_rng(50)

    # Iterate over all analysis json files
    for file in Path("./data_out/").glob("**/*analysis.json"):

        # Parse and load data
        json_data = orjson.loads(file.read_bytes())

        # Extract relevant data
        metric_config: dict[str, Any] = json_data.get("metric_config", {})
        metric_title: str = metric_config.get("title", "")
        requested_percentiles: list[float] = metric_config.get("requested_percentiles", [])
        clean: dict[str, Any] = json_data.get("clean", {})
        data: list[float] = clean.get("data", [])

        # Omit sample sizes larger than data length
        sample_sizes = list(filter(lambda x: x <= len(data), sample_sizes))

        # Initialize collected data
        collected_data: list[pd.DataFrame] = []

        # Iterate over sample sizes
        for n in sample_sizes:

            # Initialize collected data for current sample size
            collected_current_sample_size_data: list[pd.DataFrame] = []

            # Iterate over 1000 samples of current size
            for _ in range(1000):

                # randomly sample n data points
                sampled_data = rng.choice(data, size=n, replace=True).tolist()

                # Compute standardized scores
                collected_current_sample_size_data.append(
                    apply_standardization(
                        data_to_standardize=np.array(sampled_data),
                        cutoffs=json_data["bootstrap"]["cutoffs"],
                        higher_is_better=json_data["metric_config"]["higher_is_better"],
                    )
                    .drop(["standardized_value_bounds"], axis=1)
                    .loc[:, "standardized_value"]
                        .value_counts(normalize=True)
                    .rename("value")
                    .to_frame()
                    .sort_index()
                    .reset_index(drop=False, names="perc_range"))

            # Convert collected data to DataFrame
            current_sample_size_df: pd.DataFrame = (
                pd.concat(collected_current_sample_size_data)
                    .assign(
                        sample_size=n,
                        test=metric_title,
                    )
                    .loc[:, ["test", "sample_size", "perc_range", "value"]]
                )

            # append current sample size data to collected data
            collected_data.append(current_sample_size_df)

    return (
        pd.concat(collected_data)
            .groupby(["test", "sample_size", "perc_range"], as_index=False)
            .agg({"value": ["min", "max", "mean"] })
    )
