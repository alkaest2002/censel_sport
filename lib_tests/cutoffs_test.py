from pathlib import Path

import numpy as np
import orjson
import pandas as pd

from lib_analysis.utils_stats import apply_standardization


def get_cutoffs_test() -> int:
    """
    Test cutoffs.
    """

    # Define base sample_sizes
    sample_sizes = [30, 50, 100, 200, 500]

    # Initialize random generator
    rng = np.random.default_rng(50)

    for file in Path("./data_out/").glob("**/*analysis.json"):

        # Parse and load data
        json_data = orjson.loads(file.read_bytes())

        # Extract relevant data
        data = json_data["clean"]["data"]

        # Omit sample sizes larger than data length
        sample_sizes = list(filter(lambda x: x <= len(data), sample_sizes))

        # Iterate over sample sizes
        for n in sample_sizes:

            # randomly sample n data points
            sampled_data = rng.choice(data, size=n, replace=False).tolist()

            scores: pd.DataFrame = apply_standardization(
                data_to_standardize=np.array(sampled_data),
                cutoffs=json_data["bootstrap"]["cutoffs"],
                higher_is_better=json_data["metric_config"]["higher_is_better"],
            )

        print(scores)

    return 1
