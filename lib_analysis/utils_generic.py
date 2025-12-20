# mypy: disable-error-code="misc"

from pathlib import Path
from typing import Any, cast

import pandas as pd


def query_from_db(stratification: dict[str, Any] | None, db: pd.DataFrame | None = None) -> pd.DataFrame:
    """Query data from a CSV database file based on metric configuration.

    Args:
        stratification: Dictionary containing stratification or None.
        db: pandas DataFrame representing the database or None.

    Returns:
        Filtered pandas DataFrame based on the query criteria.

    Raises:
        FileNotFoundError: If the database file does not exist.
        ValueError: If the provided db is not a pandas DataFrame.
    """
    # If DataFrame is not provided, load it
    if db is None:

        # Get database file path
        filepath: Path = Path("./db") / "db.csv"

        # Check if file exists
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath.name} not found")

        # Read csv file
        db: pd.DataFrame = pd.read_csv(filepath)

    if isinstance(db, pd.DataFrame) is False:
        raise ValueError("db must be a pandas DataFrame")

    # If no filter logic provided, return full DataFrame
    if stratification is None:
        return db

    # Build db_query (i.e., filtering conditions)
    db_query: list[str] = []

    # Iterate over stratification filters
    for db_filter in stratification.values():
        # Get query part
        filter_condition: str = db_filter.get("query", "")
        # Append to db_query
        db_query.append(filter_condition)

    # Filter data with query
    return db.query(" and ".join(db_query))

def is_falsy(value: Any) -> bool:
    """Check if a value is falsy with support for numpy arrays and custom logic.

    Args:
        value: The value to check for falsiness.

    Returns:
        True if the value is falsy, False otherwise.
    """
    # Handle None explicitly
    if value is None:
        return True

    # Handle numpy arrays
    if hasattr(value, "__len__") and hasattr(value, "size"):
        try:
            return cast("bool", value.__len__() == 0 or value.size == 0)
        except (AttributeError, TypeError):
            pass

    # Handle strings specifically (including whitespace-only strings)
    if isinstance(value, str):
        return value == "" or value.isspace()

    # Handle other containers and standard falsy values
    try:
        return len(value) == 0
    except TypeError:
        # For objects without __len__, use standard truthiness
        return not bool(value)
