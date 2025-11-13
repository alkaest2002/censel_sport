from typing import TYPE_CHECKING, Any

import pandas as pd

from lib_analysis.utils_generic import is_falsy
from lib_analysis.utils_stats import apply_standardization

if TYPE_CHECKING:
    from collections.abc import Hashable

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
    data: NDArray[np.number[Any]] = clean.get("data", [])
    cutoffs = boostrap.get("cutoffs", [])

    # Raise error if something is missing
    if any(map(is_falsy, (clean, boostrap, data, cutoffs))):
        raise ValueError("---> The data dictionary does not contain all required parts.")

    # Compute standatdizatin scores
    scores: list[dict[Hashable, Any]] = (
        apply_standardization(data_to_standardize=data, cutoffs=cutoffs)
            .to_dict(orient="records")
    )

    # Compute percentage of standardized scores
    value_counts_perc = (
        pd.Series([x["standardized_value"] for x in scores])
            # Convert to int first
            .astype("int")
            # Then to string since they will be used as keys in a JSON
            .astype("string")
            # Compute value counts as percentage
            .value_counts(normalize=True, sort=False)
            # Multiply by 100 to get percentage
            .mul(100)
            # Convert to dictionary
            .to_dict()
    )

    # Update data dict
    data_dict["standardize"] = {
        "scores": scores,
        "value_counts_perc": value_counts_perc,
    }

    return data_dict
