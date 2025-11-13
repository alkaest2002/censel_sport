from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from lib_analysis.utils_generic import is_falsy

if TYPE_CHECKING:
    from numpy.typing import NDArray


def clean_data(
    data_dict: dict[str, Any],
) -> dict[str, Any]:
    """Clean performance data by removing outliers and invalid values.

    This function processes performance data by removing non-positive values, NaNs,
    and optionally outliers based on the IQR method. It also computes descriptive
    statistics and quantiles for the cleaned data.

    Args:
        data_dict: Dictionary containing performance data with the following structure:
            - "metric_config": Configuration dictionary with cleaning parameters
            - "load": Dictionary containing the raw data array
            - "data": NumPy array of performance measurements

    Returns:
        Updated data dictionary with a new "clean" key containing:
            - "data": Cleaned NumPy array
            - "descriptive_stats": Dictionary of descriptive statistics
            - "metadata": Dictionary with cleaning operation counts

    Raises:
        ValueError: If the data dictionary is missing required components
            (metric_config, load, or data).
    """
    # Extract data from dictionary
    metric_config: dict[str, Any] = data_dict.get("metric_config", {})
    load: dict[str, Any] = data_dict.get("load", {})
    data: NDArray[np.number[Any]] = load.get("data", np.array([]))

    # Raise error if something is missing
    if any(map(is_falsy, (metric_config, load, data))):
        raise ValueError("---> The data dictionary does not contain all required parts.")

    # Get cleaning parameters
    remove_outliers: bool = metric_config.get("remove_outliers", False)
    outlier_factor: float = metric_config.get("outlier_factor", 3)

    # Remove non-positive values and NaNs
    valid_mask: NDArray[np.bool_] = (data > 0) & np.isfinite(data)
    clean_data: NDArray[np.number[Any]] = data[valid_mask]

    # Count invalid
    removed_invalid: int = len(data) - len(clean_data)

    # Remove outliers if requested
    if remove_outliers:
        q1: float = np.percentile(clean_data, 25)
        q3: float = np.percentile(clean_data, 75)
        iqr: float = q3 - q1
        lower_bound: float = q1 - outlier_factor * iqr
        upper_bound: float = q3 + outlier_factor * iqr
        outlier_mask: NDArray[np.bool_] = (clean_data >= lower_bound) & (clean_data <= upper_bound)
    else:
        # No outlier method, keep everything
        outlier_mask = np.ones(len(clean_data), dtype=bool)

    removed_outliers: int = len(clean_data) - np.sum(outlier_mask).astype(int)
    final_data: NDArray[np.number[Any]] = clean_data[outlier_mask]

    # Compute descriptive statistics
    statistics: dict[Any, Any] = pd.DataFrame(final_data).describe().squeeze().to_dict() # type: ignore[union-attr]

    # Add kurtosis and skewness
    statistics["kurtosis"] = pd.Series(final_data).kurtosis()
    statistics["skewness"] = pd.Series(final_data).skew()

    # Update data dictionary
    data_dict["clean"] = {
        "data": final_data,
        "descriptive_stats": statistics,
        "metadata": {
            "removed_invalid": removed_invalid,
            "removed_outliers": removed_outliers,
            "final_size": final_data.size,
        },
    }

    return data_dict
