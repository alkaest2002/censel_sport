from typing import Any

import numpy as np


def clean_data(
    data_dict: dict[str, Any],
) -> dict[str, Any]:
    """
    Clean performance data by removing outliers and invalid values.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing data

    Returns:
    --------
    dict : Updated data dictionary with cleaned data and cleaning statistics
    """
    # Get raw data
    raw_data = data_dict.get("raw_data", np.array([]))

    # Check if raw data is empty
    if raw_data.size == 0:
        raise ValueError("No raw data found in data dictionary.")

    # Get cleaning parameters
    remove_outliers: bool = data_dict.get("remove_outliers", False)
    outlier_method: str = "iqr"
    outlier_factor: float = 1.5

    # Remove non-positive values and NaNs
    valid_mask = (raw_data > 0) & np.isfinite(raw_data)
    clean_data = raw_data[valid_mask]

    # Count invalid
    removed_invalid = len(raw_data) - len(clean_data)

    # Remove outliers if requested
    removed_outliers = 0
    if remove_outliers:
        q1 = np.percentile(clean_data, 25)
        q3 = np.percentile(clean_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - outlier_factor * iqr
        upper_bound = q3 + outlier_factor * iqr
        outlier_mask = (clean_data >= lower_bound) & (clean_data <= upper_bound)
    # No outlier method, return everything
    else:
        outlier_mask = np.ones(len(clean_data), dtype=bool)

    removed_outliers = len(clean_data) - np.sum(outlier_mask)
    analysis_data = clean_data[outlier_mask]

    # Update data dictionary
    data_dict.update({
        "analysis_data": analysis_data,
        "cleaning_stats": {
            "outlier_method": outlier_method if remove_outliers else None,
            "outlier_factor": outlier_factor if remove_outliers else None,
            "removed_invalid": removed_invalid,
            "removed_outliers": removed_outliers,
            "final_size": len(analysis_data),
        },
    })

    return data_dict
