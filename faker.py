
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from lib_analysis import MT100, MT1000, PUSHUPS, SITUPS, SWIM25
from lib_analysis.utils_stats import generate_synthetic_data

subsets = list(product(
    [2024, 2023, 2022],
    ["hd", "mlli"],
    [MT100, MT1000, SWIM25, PUSHUPS, SITUPS],
    ["M","F"],
))


# Initialize random generator
rng = np.random.default_rng(100)

# Initialize variable to hold full dataframe
df: pd.DataFrame | None = None

# Loop through all subsets and generate synthetic data
for (year, concourse, test, gender) in subsets:
    sample_size: int = rng.integers(90,150).astype(int)
    fake_data: NDArray[np.integer[Any] | np.floating[Any]]  = generate_synthetic_data(test, sample_size)
    fake_data_df = (
        pd.DataFrame({ "value": fake_data })
            .assign(
                concourse=concourse,
                gender=gender,
                year=year,
                test=test,
                age=rng.normal(loc=20, scale=1.5, size=len(fake_data)).astype(int),
            )
            .loc[:, ["year", "concourse", "test", "gender", "age", "value"]]
    )
    df = fake_data_df if df is None else pd.concat([df, fake_data_df], ignore_index=True)

# Set filepath
file_path = Path("db/synthetic_data.csv")

# Save to cb
df.to_csv(file_path, index=False) #type: ignore[union-attr]
