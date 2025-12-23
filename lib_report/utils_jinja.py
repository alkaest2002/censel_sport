import locale
from typing import Any

from lib_analysis import RECRUITMENT_TYPE, TEST


def format_number_locale(
    value: float | str | None,
    precision: int = 2,
    locale_name: str = "it_IT.UTF-8",
) -> str:
    """
    Format a number using locale-aware formatting with European conventions.

    Uses comma as decimal separator and dot as thousands separator by default.
    Raises an exception if the specified locale is not available.

    Args:
        value: The number to format. Can be int, float, string representation
               of a number, or None.
        precision: Number of decimal places to display. Defaults to 2.
        locale_name: Locale string to use for formatting. Defaults to 'de_DE.UTF-8'
                    (German locale). Other options: 'fr_FR.UTF-8', 'it_IT.UTF-8', etc.

    Returns:
        Formatted number string with European number formatting.
        Returns empty string if value is None.

    Raises:
        ValueError: If value cannot be converted to float.
        locale.Error: If the specified locale is not available on the system.
    """
    if value is None:
        return ""

    try:
        numeric_value = float(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot convert value '{value}' to float") from e

    # Store current locale to restore later
    current_locale = locale.getlocale(locale.LC_NUMERIC)

    try:
        # Set European locale
        locale.setlocale(locale.LC_NUMERIC, locale_name)

        # Format number with locale-aware formatting
        return locale.format_string(f"%.{precision}f", numeric_value, grouping=True)

    finally:
        # Always restore original locale, even if an error occurred
        locale.setlocale(locale.LC_NUMERIC, current_locale)


def format_seconds(seconds: float, precision: int, with_hours: bool = False) -> str:
    """Format seconds into a human-readable string (HH:MM:SS.sss).

    Args:
        seconds: The time in seconds to format.
        precision: Number of decimal places for fractional seconds.
        with_hours: Whether to always include hours in the output.

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
        full: str = f"{hours:02d}:{minutes:02d}:{secs:0{width}.{precision}f}".replace(".", ",")
    else:
        full = f"{hours:02d}:{minutes:02d}:{int(secs):02d}"

    # Remove hours part if not needed
    if not with_hours:
        return full[3:]

    return full

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
