# mypy: disable-error-code="misc"

from typing import Any, cast


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
