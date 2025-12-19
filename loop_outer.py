from datetime import datetime
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from lib_analysis import MT1000, RECRUITMENT_TYPE, TEST

if TYPE_CHECKING:
    import builtins


def main() -> int:  # noqa: PLR0911
    """
    Run the outer pipeline.

    Returns:
    --------
    int
        Process exit status code:
        - 0: Success
        - 1: Error in file validation or loading
    """
    # Define file paths
    db_original_filepath: Path = Path("db/db_original2.csv")
    db_filepath: Path = Path("db/db.csv")

    # Read CSV file into DataFrame
    db: pd.DataFrame = pd.read_csv(db_original_filepath)

    ######################################################################
    # Basic Sanity checks
    # Ensure the DataFrame has the expected structure data types
    # And sensible values
    # Does not deal with outliers or deeper data quality issues
    ######################################################################

    # Year column
    conditions: list[bool | np.bool[builtins.bool]] = [
        "recruitment_year" in db.columns,
        db["recruitment_year"].notna().all(),
        db["recruitment_year"].dtype == "int64",
        db["recruitment_year"].between(2021, datetime.now(ZoneInfo("Europe/Rome")).year).all(),
    ]

    # Raise error if any condition fails
    if not all(conditions):
        print("FAILED Year checks")
        return 1

    # Recruitment Type column
    conditions = [
        "recruitment_type" in db.columns,
        db["recruitment_type"].notna().all(),
        db["recruitment_type"].dtype == "object",
        db["recruitment_type"].isin(list(RECRUITMENT_TYPE.keys())).all(),
    ]

    # Raise error if any condition fails
    if not all(conditions):
        print("FAILED Concourse checks")
        return 1

    # Test column
    conditions = [
        "test" in db.columns,
        db["test"].notna().all(),
        db["test"].dtype == "object",
        db["test"].isin(list(TEST.keys())).all(),
    ]

    # Raise error if any condition fails
    if not all(conditions):
        print("FAILED Test checks")
        return 1

    # Gender column
    conditions = [
        "gender" in db.columns,
        db["gender"].notna().all(),
        db["gender"].dtype == "object",
        db["gender"].isin(["M", "F"]).all(),
    ]

    # Raise error if any condition fails
    if not all(conditions):
        print("FAILED Gender checks")
        return 1

    # Age column
    conditions = [
        "age" in db.columns,
        db["age"].notna().all(),
        db["age"].dtype == "int64",
        db["age"].between(17, 28).all(),
    ]

    # Raise error if any condition failsß
    if not all(conditions):
        print("FAILED Age checks")
        return 1

    # Value column
    conditions = [
        "value" in db.columns,
        db["value"].notna().all(),
        db["value"].dtype == "float64",
        np.isfinite(db["value"]).all(),
        (db["value"] >= 0).all(),
    ]

    # Raise error if any condition fails
    if not all(conditions):
        print("FAILED Value checks")
        return 1

    print("DB Data validated successfully.")


    # ##############################################################################
    # In italian: elimina dal db tutte le prove obblicatorie a sbarramento
    # ##############################################################################

    # Base conditions
    c_mt1000: pd.Series = db["test"] == MT1000

    # Drop year 2021 for all tests
    drop_year_2021: pd.Series = db["recruitment_year"].eq(2021)

    # Drop MT1000 for all years before 2023
    drop_mt_1000: pd.Series = c_mt1000 & db["recruitment_year"].lt(2023)

    # Combine all drop conditions
    drop: pd.Series = (drop_year_2021 | drop_mt_1000)

    # Drop rows matching the conditions
    filterd_db: pd.DataFrame = db[~drop]

    # Save filtered DataFrame to a new CSV file
    filterd_db.to_csv(db_filepath, index=False)

    return 0

if __name__ == "__main__":
    sys.exit(main())

