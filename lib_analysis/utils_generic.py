# mypy: disable-error-code="misc"

from pathlib import Path
from typing import Any, cast

import pandas as pd


def query_from_db(metric_config: dict[str, Any]) -> pd.DataFrame:
    """Query data from a CSV database file based on metric configuration.

    Args:
        metric_config: Dictionary containing metric configuration.

    Returns:
        Filtered pandas DataFrame based on the query criteria.

    Raises:
        FileNotFoundError: If the database file does not exist.
    """
    # If file does not exist
    filepath: Path = Path("./db") / "db.csv"

    # Check if file exists
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath.name} not found")

    # Read csv file
    df: pd.DataFrame = pd.read_csv(filepath)

    # Extract data from metric_config
    metric_id: str = "_".join(metric_config["id"].split("_")[:2])
    stratification: dict[str, Any] = metric_config.get("stratification", {})

    # Build db_query
    db_query: str = f"test=='{metric_id}'"

    # Add stratification filters to db_query
    db_filter: dict[str, Any]

    # Iterate over stratification filters
    for db_filter in stratification.values():
        # Get query part
        filter_condition: str = db_filter.get("query", "")
        # Append to db_query
        db_query += f" and {filter_condition}" if filter_condition else ""

    # Filter data with query
    return df.query(db_query)


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


def format_seconds(seconds: float) -> str:
    """Format seconds into a human-readable string (HH:MM:SS.sss).

    Args:
        seconds: The time in seconds to format.

    Returns:
        Formatted time string in HH:MM:SS.sss format.
    """
    # Separate whole seconds and fractional part
    whole_seconds: int = int(seconds)
    fractional_part: float = seconds - whole_seconds

    # Use divmod to get hours, minutes, and seconds
    hours: int
    remainder: int
    hours, remainder = divmod(whole_seconds, 3600)

    minutes: int
    secs: int
    minutes, secs = divmod(remainder, 60)

    # Format with fractional seconds
    return f"{hours:02d}:{minutes:02d}:{secs + fractional_part:05.2f}"


def format_title(title: str) -> str:
    """Format a title string by capitalizing the first letter and lowercasing the rest.

    Args:
        title: The title string to format.

    Returns:
        Formatted title string with first letter capitalized and rest lowercase.
        Returns empty string if input is empty.
    """
    return title[0].upper() + title[1:].lower() if title else ""
