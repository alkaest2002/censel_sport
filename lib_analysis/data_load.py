from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from lib_analysis.utils_generic import query_from_db
from lib_analysis.utils_stats import generate_synthetic_data


def _get_descriptive_stats(data: NDArray[np.number[Any]]) -> pd.DataFrame:
    """Compute descriptive statistics for the given data.

    Args:
        data: Numpy array of performance data.

    Returns:
         pd.DataFrame: dataframe containing descriptive statistics.
    """
    return pd.DataFrame(data).describe()

def _load_from_db(metric_config: dict[str, Any]) -> dict[str, Any]:
    """Load performance data from a CSV file.

    Args:
        metric_config: Configuration dictionary containing metric details such as
            id, units, higher_is_better, and optional stratification parameters.

    Returns:
        Dictionary containing:
            - metric_config: The input configuration
            - load: Dictionary with data and metadata

    Raises:
        FileNotFoundError: If the database CSV file does not exist.
    """

    try:
        # Load data from database
        db_df: pd.DataFrame = query_from_db(metric_config)

        # Enforce value column to be numeric
        raw_data: np.ndarray = pd.to_numeric(db_df.loc[:, "value"], downcast="integer").to_numpy()

        # Generate descriptive statistics from data
        descriptive_stats: pd.DataFrame = _get_descriptive_stats(raw_data)

    # Catch exceptions
    except Exception as e:  # noqa: BLE001
        print(e)
        return {
            "metric_config": metric_config,
            "query_from_db": None,
            "load": {
                "data": np.array([]),
                "descriptive_stats": {},
                "metadata": {
                    "original_size": 0,
                    "valid_records": 0,
                    "error": str(e),
                },
            },
        }
    else:
        return {
            "metric_config": metric_config,
            "query_from_db": db_df,
            "load": {
                "data": raw_data,
                "descriptive_stats": descriptive_stats,
                "metadata": {
                    "original_size": len(db_df),
                    "valid_records": len(raw_data),
                    "error": None,
                },
            },
        }


def _load_from_synthetic(metric_config: dict[str, Any]) -> dict[str, Any]:
    """Load performance data from synthetic sources.

    Args:
        metric_config: Configuration dictionary containing:
            - id: Metric identifier
            - synthetic_n_samples: Number of samples to generate (default: 300)
            - random_state: Random seed for reproducibility (default: 42)

    Returns:
        Dictionary containing:
            - metric_config: The input configuration
            - load: Dictionary with generated data, descriptive statistics,
              quantiles, and metadata
    """
    # Get number of samples to generate
    n_samples: int = metric_config.get("synthetic_n_samples", 300)

    # Set random seed for reproducibility
    random_state: int = metric_config.get("random_state", 42)

    # Get metric id, Keep only first suffix (denoted by underscore)
    metric_id: str = "_".join(metric_config["id"].split("_")[:2])

    # Generate synthetic data
    raw_data: NDArray[np.number[Any]] = generate_synthetic_data(metric_id, n_samples, random_state)

    # Generate descriptive statistics from data
    descriptive_stats: pd.DataFrame = _get_descriptive_stats(raw_data)

    return {
        "metric_config": metric_config,
        "query_from_db": None,
        "load": {
            "data": raw_data,
            "descriptive_stats": descriptive_stats,
            "metadata": {
                "original_size": len(raw_data),
                "valid_records": len(raw_data),
                "error": None,
            },
        },
    }


def load_data(metric_config: dict[str, Any]) -> dict[str, Any]:
    """Load performance data based on the metric configuration.

    Args:
        metric_config: Configuration dictionary that must contain a 'source_type'
            key with value either 'db' or 'synthetic'. Additional keys depend on
            the source type selected.

    Returns:
        Dictionary containing the loaded data, metric configuration, and metadata.
        Structure matches the return format of the specific loader function used.

    Raises:
        NotImplementedError: If source_type is not 'db' or 'synthetic'.
    """
    # Determine source type and load data accordingly
    source_type: str = metric_config.get("source_type", "")

    if source_type not in ["db", "synthetic"]:
        raise NotImplementedError(f"---> Unknown source_type '{source_type}' in metric configuration.")

    return (
        _load_from_db(metric_config) if source_type == "db"
            else _load_from_synthetic(metric_config)
    )
