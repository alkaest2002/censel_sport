from pathlib import Path
from typing import Any

import pandas as pd

from lib.utils import generate_synthetic_data


def load_from_csv(metric_config: dict[str, Any]) -> dict[str, Any]:
    """
    Load performance data from CSV file.

    Parameters:
    -----------
    metric_config : dict
        Configuration for the metric (name, units, higher_is_better)

    Returns:
    --------
    dict : Contains raw_data, metric_config, and metadata
    """
    # If file does not exist
    filepath = Path("./data_in") / metric_config["source_filename"]

    # Check if file exists
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath.name} not found")
    try:
        # Read csv file
        df = pd.read_csv(filepath)

        # Enforce data to be numeric
        raw_data = pd.to_numeric(df, downcast="integer").to_numpy()

    # Catch exceptions
    except Exception as e:  # noqa: BLE001
        print(e)
        return {
            "metric_config": metric_config,
            "raw_data": None,
            "metadata": {
                "original_size": 0,
                "valid_records": 0,
                "error": str(e),
            },
        }
    else:
        return {
            "metric_config": metric_config,
            "raw_data": raw_data,
            "metadata": {
                "original_size": len(df),
                "valid_records": len(raw_data),
                "error": None,
            },
        }


def load_from_synthetic(metric_config: dict[str, Any]) -> dict[str, Any]:
    """
    Generate synthetic data for testing.

    Parameters:
    -----------
    metric_config : dict
        Configuration for the metric

    Returns:
    --------
    dict : Contains raw_data, metric_config, and metadata
    """
    # Get number of samples to generate
    n_samples = metric_config.get("synthetic_n_samples", 500)

    # Generate synthetic data
    raw_data = generate_synthetic_data(metric_config["name"], n_samples)

    return {
        "metric_config": metric_config,
        "raw_data": raw_data,
        "metadata": {
            "original_size": len(raw_data),
            "valid_records": len(raw_data),
            "error": None,
        },
    }
