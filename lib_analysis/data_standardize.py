from typing import TYPE_CHECKING, Any

import pandas as pd

from lib_analysis.utils_generic import is_falsy
from lib_analysis.utils_stats import apply_standardization

if TYPE_CHECKING:
    from collections.abc import Hashable

    import numpy as np
    from numpy.typing import NDArray


def compute_standard_scores(data_dict: dict[str, Any]) -> dict[str, Any]:
    """Compute standardized scores from analysis data using normative cutoffs.

    Takes a data dictionary containing clean data and bootstrap cutoffs,
    applies standardization to compute standard scores, and calculates
    percentage distributions of standardized values.

    Args:
        data_dict: Dictionary containing analysis data with the following
            expected structure:
            - "clean": Dict containing "data" key with numerical array
            - "bootstrap": Dict containing "cutoffs" key with cutoff values

    Returns:
        Updated data dictionary with added "standardize" key containing:
        - "scores": List of dictionaries with standardized scores
        - "value_counts_perc": Dictionary with percentage distribution of
          standardized values

    Raises:
        ValueError: If the data dictionary is missing required components
            (clean, bootstrap, data, or cutoffs).
    """
    # Extract data from dictionary
    clean: dict[str, Any] = data_dict.get("clean", {})
    bootstrap: dict[str, Any] = data_dict.get("bootstrap", {})
    data: NDArray[np.number[Any]] = clean.get("data", [])
    cutoffs: list[tuple[float, float]] = bootstrap.get("cutoffs", [])

    # Raise error if something is missing
    if any(map(is_falsy, (clean, bootstrap, data, cutoffs))):
        raise ValueError("The data dictionary does not contain all required parts.")

    # Compute standardization scores
    scores: list[dict[Hashable, Any]] = (
        apply_standardization(data_to_standardize=data, cutoffs=cutoffs)
            .to_dict(orient="records")
    )

    # Compute percentage of standardized scores
    value_counts_perc: dict[str, float] = (
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
