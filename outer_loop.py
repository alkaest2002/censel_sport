from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

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
    filepath = Path("db/db.csv")

    # Read CSV file into DataFrame
    df = pd.read_csv(filepath)

    ######################################################################
    # Basic Sanity checks
    # Ensure the DataFrame has the expected structure data types
    # And sensible values
    # Does not deal with outliers or deeper data quality issues
    ######################################################################

    # Year column
    conditions = [
        "recruitment_year" in df.columns,
        df["recruitment_year"].notna().all(),
        df["recruitment_year"].dtype == "int64",
        df["recruitment_year"].between(2000, datetime.now(ZoneInfo("Europe/Rome")).year).all(),
    ]
    if not all(conditions):
        print("FAILED Year checks")
        return 1

    # Recruitment Type column
    conditions = [
        "recruitment_type" in df.columns,
        df["recruitment_type"].notna().all(),
        df["recruitment_type"].dtype == "object",
        df["recruitment_type"].isin(["hd", "mlli"]).all(),
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
        df["gender"].isin(["M", "F"]).all(),
    ]
    if not all(conditions):
        print("FAILED Gender checks")
        return 1

    # Age column
    conditions = [
        "age" in df.columns,
        df["age"].notna().all(),
        df["age"].dtype == "int64",
        df["age"].between(14, 99).all(),
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

    # Compute percentage of duplicates
    duplicated: float = round((df.duplicated().sum() / df.shape[0]) * (100), 2)

    print(f"Data loaded and validated successfully. Percentage of duplicates {duplicated}%")
    return 0

if __name__ == "__main__":
    sys.exit(main())

