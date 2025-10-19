from typing import Any, Literal

import numpy as np
from scipy import stats


def clean_data(
    data_dict: dict[str, Any],
    remove_outliers: bool=False,
    outlier_method: Literal["iqr", "zscore"]="iqr",
    outlier_factor: float=4,
) -> dict[str, Any]:
    """
    Clean performance data by removing outliers and invalid values.

    Parameters:
    -----------
    data_dict : dict
        Data dictionary from DataLoader
    remove_outliers : bool
        Whether to remove outliers
    outlier_method : str
        Method for outlier detection ('iqr' or 'zscore')
    outlier_factor : float
        Factor for outlier detection

    Returns:
    --------
    dict : Updated data dictionary with cleaned data
    """
    # Get raw data
    raw_data = data_dict["raw_data"]

    # Remove non-positive values and NaNs
    valid_mask = (raw_data > 0) & np.isfinite(raw_data)
    clean_data = raw_data[valid_mask]

    # Count invalid
    removed_invalid = len(raw_data) - len(clean_data)

    # Remove outliers if requested
    removed_outliers = 0
    if remove_outliers:
        # IQR outlier method
        if outlier_method == "iqr":
            q1 = np.percentile(clean_data, 25)
            q3 = np.percentile(clean_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - outlier_factor * iqr
            upper_bound = q3 + outlier_factor * iqr
            outlier_mask = (clean_data >= lower_bound) & (clean_data <= upper_bound)
        # ZScore outlier method
        elif outlier_method == "zscore":
            z_scores = np.abs(stats.zscore(clean_data))
            outlier_mask = z_scores <= outlier_factor
    # No outlier method, return everything
    else:
        outlier_mask = np.ones(len(clean_data), dtype=bool)

    removed_outliers = len(clean_data) - np.sum(outlier_mask)
    analysis_data = clean_data[outlier_mask]

    # Update data dictionary
    data_dict.update({
        "analysis_data": analysis_data,
        "cleaning_stats": {
            "removed_invalid": removed_invalid,
            "removed_outliers": removed_outliers,
            "final_size": len(clean_data),
            "outlier_method": outlier_method if remove_outliers else "none",
        },
    })

    return data_dict
