from pathlib import Path
import sys

import pandas as pd

from lib_analysis import MT100, MT1000, PUSHUPS, SITUPS, SWIM25


def main() -> int:  # noqa: PLR0911
    """
    Run the full data analysis pipeline.

    Returns:
    --------
    int
        Process exit status code:
        - 0: Success
        - 1: Error in file validation or loading
    """
    # Define file path
    filepath = Path("db/synthetic_data.csv")

    # Read CSV file into DataFrame
    df = pd.read_csv(filepath)

    # Expected columns --- > "year", "concourse", "test", "gender", "age", "value"

    ################################################################
    # Sanity checks
    ################################################################

    # Year column
    conditions = [
        "year" in df.columns,
        df["year"].notna().all(),
        df["year"].dtype == "int64",
        df["year"].between(2022, 2024).all(),
    ]
    if not all(conditions):
        print("FAILED Year checks")
        return 1

    # Concourse column
    conditions = [
        "concourse" in df.columns,
        df["concourse"].notna().all(),
        df["concourse"].dtype == "object",
        df["concourse"].isin(["hd", "mlli"]).all(),
    ]
    if not all(conditions):
        print("FAILED Concourse checks")
        return 1

    # Test column
    conditions = [
        "test" in df.columns,
        df["test"].notna().all(),
        df["test"].dtype == "object",
        df["test"].isin([MT100, MT1000, SWIM25, SITUPS, PUSHUPS]).all(),
    ]
    if not all(conditions):
        print("FAILED Test checks")
        return 1

    # Gender column
    conditions = [
        "gender" in df.columns,
        df["gender"].notna().all(),
        df["gender"].dtype == "object",
        df["gender"].str.lower().isin(["m", "f"]).all(),
    ]
    if not all(conditions):
        print("FAILED Gender checks")
        return 1

    # Age column
    conditions = [
        "age" in df.columns,
        df["age"].notna().all(),
        df["age"].dtype == "int64",
        df["age"].between(17, 30).all(),
    ]
    if not all(conditions):
        print("FAILED Age checks")
        return 1

    # Value column
    conditions = [
        "value" in df.columns,
        df["value"].notna().all(),
        df["value"].dtype == "float64",
        (df["value"] >= 0).all(),
    ]
    if not all(conditions):
        print("FAILED Value checks")
        return 1

    print("Data loaded and validated successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

