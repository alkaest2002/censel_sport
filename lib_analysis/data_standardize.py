from typing import TYPE_CHECKING, Any

from lib_analysis.utils_generic import is_falsy
from lib_analysis.utils_stats import apply_standardization

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    import pandas as pd


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
    metric_config: dict[str, Any] = data_dict.get("metric_config", {})
    clean: dict[str, Any] = data_dict.get("clean", {})
    data: NDArray[np.number[Any]] = clean.get("data", [])
    bootstrap: dict[str, Any] = data_dict.get("bootstrap", {})
    cutoffs: list[tuple[float, float]] = bootstrap.get("cutoffs", [])
    awarded_scores: list[float] = metric_config.get("awarded_scores", [])
    higher_is_better: bool = metric_config.get("higher_is_better", False)

    # Raise error if something is missing
    if any(map(is_falsy, (metric_config, clean, data, bootstrap, cutoffs, awarded_scores))):
        raise ValueError("---> The data dictionary does not contain all required parts.")

    # Compute standardization scores
    standardized_scores: pd.DataFrame = (
        apply_standardization(
            data_to_standardize=data,
            cutoffs=cutoffs,
            awarded_scores=awarded_scores,
            higher_is_better=higher_is_better,
        )
    )

    # Compute percentage of standardized scores
    value_counts_perc: pd.Series = (
        standardized_scores["standardized_step"]
            # Compute value counts as percentage
            .value_counts(normalize=True, sort=False)
            # Reindex to ensure all steps are present
            .reindex(range(1, len(cutoffs)+1), fill_value=0)
            # Multiply by 100 to get percentage
            .mul(100)
    )

    # Make sure value_counts_perc index is of string type
    value_counts_perc.index =  value_counts_perc.index.astype(str)

    # Update data dict
    data_dict["standardize"] = {
        "scores": standardized_scores.to_dict(orient="records"),
        "step_perc": value_counts_perc.to_dict(),
    }

    return data_dict
