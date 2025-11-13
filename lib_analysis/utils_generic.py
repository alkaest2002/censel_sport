# mypy: disable-error-code="misc"

from typing import Any, cast


def is_falsy(value: Any) -> bool:
    """Check if a value is falsy with support for numpy arrays and custom logic.

    Args:
        value: The value to check.

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
    whole_seconds = int(seconds)
    fractional_part = seconds - whole_seconds

    # Use divmod to get hours, minutes, and seconds
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    # Format with fractional seconds
    return f"{hours:02d}:{minutes:02d}:{secs + fractional_part:05.2f}"
