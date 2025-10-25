# mypy: disable-error-code="misc"

import argparse
from pathlib import Path
from typing import Any, Literal, cast

import orjson


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Parameters:
    -----------
    None

    Returns:
    --------
    argparse.Namespace
        Parsed arguments with attribute:
        - filepath (str): Path to the data file to report.
    """
    parser = argparse.ArgumentParser(
        description="Generate statistical analysis and reporting from data file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the data file to analyze/report",
    )

    return parser.parse_args()


def validate_file_path(filepath: str, type_of_request: Literal["analysis", "report"]) -> Path:
    """
    Validate that the file path exists, points to a regular file, and is non-empty.

    Parameters:
    -----------
    filepath : str
        Path to the file to validate.

    type_of_request : Literal["analysis", "report "]
        Type of file to validate, either "analysis" or "report ".

    Returns:
    --------
    Path
        pathlib.Path object for the validated file.

    Raises:
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the path is not a file or the file is empty.
    """
    file_path = (
        Path("./data_out") / filepath / f"{filepath}_analysis.json" if type_of_request == "report"
        else Path("./data_in") / f"{Path(filepath)}.json"
    )
    print(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {filepath}")

    if not file_path.stat().st_size > 0:
        raise ValueError(f"File is empty: {filepath}")

    return file_path


def load_configuration_data(filepath: Path) -> Any:
    """
    Load JSON data from file.

    Parameters:
    -----------
    filepath : Path
        Path to the JSON file containing report input data.

    Returns:
    --------
    dict
        Parsed JSON object as a dictionary.

    Raises:
    ------
    OSError
        If the file cannot be opened or read.
    orjson.JSONDecodeError
        If the file content is not valid JSON.
    """
    with filepath.open("rb") as fin:
        return orjson.loads(fin.read())


def is_falsy(value: Any) -> bool:
    """
    Check if a value is falsy with support for numpy arrays and custom logic.

    Parameters
    ----------
    value : Any
        The value to check

    Returns:
    -------
    bool
        True if the value is falsy, False otherwise
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
