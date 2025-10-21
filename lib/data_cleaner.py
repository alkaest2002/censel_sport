from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
    # Extract from data_dict
    metric_config =  data_dict.get("metric_config", {})
    load: dict[str, Any] = data_dict.get("load", {})
    data: NDArray[np.integer[Any] | np.floating[Any]] = load.get("data", np.array([]))

    # Check if raw data is empty
    if data.size == 0:
        raise ValueError("No raw data found in data dictionary.")

    # Get cleaning parameters
    remove_outliers: bool = metric_config.get("remove_outliers", False)
    outlier_factor: float = metric_config.get("outlier_factor", 1.5)

    # Remove non-positive values and NaNs
    valid_mask = (data > 0) & np.isfinite(data)
    clean_data = data[valid_mask]

    # Count invalid
    removed_invalid = len(data) - len(clean_data)

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
    final_data = clean_data[outlier_mask]

    # Update data dictionary
    data_dict["clean"] = ({
        "data": final_data,
        "metadata": {
            "removed_invalid": removed_invalid,
            "removed_outliers": removed_outliers,
            "final_size": final_data.size,
        },
    })

    return data_dict
