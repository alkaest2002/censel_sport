"""Generate synthetic data for military fitness tests and save to CSV.

This module creates fake data for various military fitness tests across different
recruitment years, types, and demographics, then saves the combined dataset to a CSV file.
Justification: This module is intended to test data analysis and reporting functionalities
by providing a comprehensive synthetic dataset.
"""

from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from lib_analysis import MT100, MT1000, PUSHUPS, SITUPS, SWIM25
from lib_analysis.utils_stats import generate_synthetic_data

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _generate_fake_military_data() -> pd.DataFrame:
    """Generate synthetic military fitness test data for multiple subsets.

    Creates fake data for all combinations of recruitment years (2022-2024),
    recruitment types (hd, mlli), fitness tests, and genders. Each subset
    contains 300 samples with ages normally distributed around 20 years.

    Returns:
        pd.DataFrame: Combined dataframe containing synthetic data with columns:
            - recruitment_year: Year of recruitment (2022, 2023, or 2024)
            - recruitment_type: Type of recruitment ("hd" or "mlli")
            - test: Fitness test type (MT100, MT1000, SWIM25, PUSHUPS, or SITUPS)
            - gender: Gender ("M" or "F")
            - age: Age (integer, normally distributed around 20)
            - value: Test performance value (synthetic)
    """
    # Define all possible combinations of parameters
    subsets: list[tuple[int, str, str, str]] = list(product(
        [2024, 2023, 2022],
        ["hd", "mlli"],
        [MT100, MT1000, SWIM25, PUSHUPS, SITUPS],
        ["M", "F"],
    ))

    # Initialize random generator with fixed seed for reproducibility
    rng: np.random.Generator = np.random.default_rng(100)

    # Intialize list to collect dataframes
    dataframes: list[pd.DataFrame] = []

    # Loop through all subsets and generate synthetic data
    for (recruitment_year, recruitment_type, test, gender) in subsets:
        fake_data: NDArray[np.number[Any]] = generate_synthetic_data(test, 300, 50)
        fake_data_df = (
            pd.DataFrame({"value": fake_data})
            .assign(
                recruitment_year=recruitment_year,
                recruitment_type=recruitment_type,
                test=test,
                gender=gender,
                age=rng.normal(loc=20, scale=1.5, size=len(fake_data)).astype(int),
            )
            .loc[:, ["recruitment_year", "recruitment_type", "test", "gender", "age", "value"]]
        )
        dataframes.append(fake_data_df)

    return pd.concat(dataframes, ignore_index=True)


def _save_data_to_csv(df: pd.DataFrame, file_path: Path) -> None:
    """Save dataframe to CSV file.

    Args:
        df: DataFrame to save.
        file_path: Path where the CSV file should be saved.
    """
    df.to_csv(file_path, index=False)


def main() -> None:
    """Main function to generate synthetic data and save to CSV file."""
    # Generate synthetic military fitness data
    df: pd.DataFrame = _generate_fake_military_data()

    # Set filepath
    file_path: Path = Path("db/db.csv")

    # Save to CSV
    _save_data_to_csv(df, file_path)


if __name__ == "__main__":
    main()
