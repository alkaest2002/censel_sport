from typing import TYPE_CHECKING, Any

from lib.utils import apply_standardization

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
    # Extract from data dictionary
    clean: dict[str, Any] = data_dict.get("clean", {})
    data: NDArray[np.integer[Any] | np.floating[Any]] = clean.get("data", [])
    boostrap: dict[str, Any] = data_dict["bootstrap"]
    cutoffs = boostrap.get("cutoffs", [])

    # Update data dict
    data_dict["standardize"] = {
        "scores": apply_standardization(data_to_standardize=data, cutoffs=cutoffs),
    }

    return data_dict
