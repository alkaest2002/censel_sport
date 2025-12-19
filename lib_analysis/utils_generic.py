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


def format_seconds(seconds: float, precision: int) -> str:
    """Format seconds into a human-readable string (HH:MM:SS.sss).

    Args:
        seconds: The time in seconds to format.
        precision: Number of decimal places for fractional seconds.

    Returns:
        Formatted time string in HH:MM:SS.sss format.
    """
    # Calculate hours, minutes, and seconds
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    # Format the result with proper width for seconds
    if precision > 0:
        width = 3 + precision  # 2 digits + decimal point + precision digits
        return f"{hours:02d}:{minutes:02d}:{secs:0{width}.{precision}f}"

    return f"{hours:02d}:{minutes:02d}:{int(secs):02d}"


def format_title(title: str) -> str:
    """Format a title string by capitalizing the first letter and lowercasing the rest.

    Args:
        title: The title string to format.

    Returns:
        Formatted title string with first letter capitalized and rest lowercase.
        Returns empty string if input is empty.
    """
    return title[0].upper() + title[1:].lower() if title else ""
