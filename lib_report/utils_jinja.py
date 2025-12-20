from typing import Any

from lib_analysis import RECRUITMENT_TYPE, TEST


def format_seconds(seconds: float, precision: int) -> str:
    """Format seconds into a human-readable string (HH:MM:SS.sss).

    Args:
        seconds: The time in seconds to format.
        precision: Number of decimal places for fractional seconds.

    Returns:
        Formatted time string in HH:MM:SS.sss format.
    """
    # Calculate hours, minutes, and seconds
    hours: int = int(seconds // 3600)
    minutes: int = int((seconds % 3600) // 60)
    secs: float = seconds % 60

    # Format the result with proper width for seconds
    if precision > 0:
        width: int = 3 + precision  # 2 digits + decimal point + precision digits
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

def get_test_label(x: dict[str, Any]) -> any:
    """Get the label for a test from the TEST mapping.

    Args:
        x: The test identifier.

    Returns:
        The corresponding label from the TEST mapping, or the input if not found.
    """
    return TEST.get(x, x)

def get_recruitment_type_label(x: dict[str, Any]) -> any:
    """Get the label for a recruitment type from the RECRUITMENT_TYPE mapping.

    Args:
        x: The recruitment type identifier.

    Returns:
        The corresponding label from the RECRUITMENT_TYPE mapping, or the input if not found.
    """
    return RECRUITMENT_TYPE.get(x, x)
