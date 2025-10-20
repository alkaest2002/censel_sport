from typing import Any

from lib.utils import apply_standardization


def compute_standard_scores(data_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Compute standardized scores from analysis data using normative cutoffs.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing data

    Returns:
    --------
    dict : Updated data dictionary with standardized scores
    """
    data_to_standardize = data_dict["analysis_data"]
    cutoffs = data_dict.get("normative_table", {}).get("computed_cutoffs", [])

    data_dict["standardized"] = (apply_standardization(
        data_to_standardize=data_to_standardize,
        cutoffs=cutoffs,
    ))

    return data_dict
