from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from lib_analysis.utils_generic import is_falsy

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
    dict : Updated data dictionary
    """
    # Extract data from dictionary
    metric_config: dict[str, Any] =  data_dict.get("metric_config", {})
    load: dict[str, Any] = data_dict.get("load", {})
    data: NDArray[np.integer[Any] | np.floating[Any]] = load.get("data", np.array([]))

    # Raise error if something is missing
    if any(map(is_falsy, (metric_config, load, data))):
        raise ValueError("---> The data dictionary does not contain all required parts.")

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
    # No outlier method, keep everything
    else:
        outlier_mask = np.ones(len(clean_data), dtype=bool)

    removed_outliers = len(clean_data) - np.sum(outlier_mask)
    final_data = clean_data[outlier_mask]

    # Statistics and quantiles
    statistics = pd.DataFrame(final_data).describe().squeeze()

    # Add kurtosis and skewness
    statistics["kurtosis"] = pd.Series(final_data).kurtosis() # type: ignore[index]
    statistics["skewness"] = pd.Series(final_data).skew() # type: ignore[index]

    # Update data dictionary
    data_dict["clean"] = ({
        "data": final_data,
        "quantiles": {
            f"q{int(q*100)}": cast("float", np.quantile(final_data, q)) for q in np.arange(0.01, 1., 0.01)
        },
        "descriptive_stats": statistics.to_dict(), # type: ignore[union-attr]
        "metadata": {
            "removed_invalid": removed_invalid,
            "removed_outliers": removed_outliers,
            "final_size": final_data.size,
        },
    })

    return data_dict
