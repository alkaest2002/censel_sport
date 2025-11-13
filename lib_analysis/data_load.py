from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from lib_analysis.utils_stats import generate_synthetic_data


def _load_from_db(metric_config: dict[str, Any]) -> dict[str, Any]:
    """Load performance data from a CSV file.

    Args:
        metric_config: Configuration dictionary containing metric details such as
            id, units, higher_is_better, and optional stratification parameters.

    Returns:
        Dictionary containing:
            - metric_config: The input configuration
            - load: Dictionary with data, quantiles, and metadata

    Raises:
        FileNotFoundError: If the database CSV file does not exist.
    """
    # If file does not exist
    filepath = Path("./db") / "db.csv"

    # Check if file exists
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath.name} not found")

    # Extract data from metric_config
    metric_id: str = "_".join(metric_config["id"].split("_")[:2])
    stratification = metric_config.get("stratification", {})

    try:
        # Read csv file
        df = pd.read_csv(filepath)

        # Build db_query
        db_query: str = f"test=='{metric_id}'"
        for db_filter in stratification.values():
            _, query = db_filter.values()
            db_query += f" and {query}" if query else ""

        # Filter data with query
        df_query = df.query(db_query)
        # Enforce data to be numeric
        raw_data = pd.to_numeric(df_query.loc[:, "value"], downcast="integer").to_numpy()

    # Catch exceptions
    except Exception as e:  # noqa: BLE001
        print(e)
        return {
            "metric_config": metric_config,
            "load": {
                "data": None,
                "quantiles": None,
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
            "load": {
                "data": raw_data,
                "quantiles": None,
                "metadata": {
                    "original_size": len(df),
                    "valid_records": len(raw_data),
                    "error": None,
                },
            },
        }


def _load_from_synthetic(metric_config: dict[str, Any]) -> dict[str, Any]:
    """Load performance data from synthetic sources.

    Generates synthetic performance data based on the metric configuration.
    Uses the metric ID to determine the data generation pattern and applies
    the specified random seed for reproducibility.

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
    n_samples = metric_config.get("synthetic_n_samples", 300)

    # Set random seed for reproducibility
    random_state = metric_config.get("random_state", 42)

    # Get metric id, Keep only first suffix (denoted by underscore)
    metric_id = "_".join(metric_config["id"].split("_")[:2])

    # Generate synthetic data
    raw_data = generate_synthetic_data(metric_id, n_samples, random_state)

    return {
        "metric_config": metric_config,
        "load": {
            "data": raw_data,
            "descriptive_stats": pd.DataFrame(raw_data).describe().squeeze().to_dict(),  # type: ignore[union-attr]
            "quantiles": {
                f"q{int(q*100)}": cast("float", np.quantile(raw_data, q)) for q in np.arange(0.01, 1., 0.01)
            },
            "metadata": {
                "original_size": len(raw_data),
                "valid_records": len(raw_data),
                "error": None,
            },
        },
    }


def load_data(metric_config: dict[str, Any]) -> dict[str, Any]:
    """Load performance data based on the metric configuration.

    Dispatches data loading to the appropriate function based on the source_type
    specified in the metric configuration. Supports loading from database files
    or generating synthetic data.

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
    source_type = metric_config.get("source_type")

    if source_type not in ["db", "synthetic"]:
        raise NotImplementedError(f"---> Unknown source_type {source_type} in metric configuration.")

    return (
        _load_from_db(metric_config) if source_type == "db"
        else _load_from_synthetic(metric_config)
    )
