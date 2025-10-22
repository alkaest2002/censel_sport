from typing import TYPE_CHECKING, Any

from lib.utils import apply_standardization, is_falsy

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def compute_standard_scores(data_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Compute standardized scores from analysis data using normative cutoffs.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing data

    Returns:
    --------
    dict : Updated data dictionary
    """
    # Extract data from dictionary
    clean: dict[str, Any] = data_dict.get("clean", {})
    boostrap: dict[str, Any] = data_dict.get("bootstrap", [])
    data: NDArray[np.integer[Any] | np.floating[Any]] = clean.get("data", [])
    cutoffs = boostrap.get("cutoffs", [])

    # Raise error if something is missing
    if any(map(is_falsy, (clean, boostrap, data, cutoffs))):
        raise ValueError("The data dictionary does not contain all required parts.")

    # Update data dict
    data_dict["standardize"] = {
        "scores": apply_standardization(data_to_standardize=data, cutoffs=cutoffs),
    }

    return data_dict
