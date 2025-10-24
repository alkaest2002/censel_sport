from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from lib.utils_stats import generate_synthetic_data


def _load_from_csv(metric_config: dict[str, Any]) -> dict[str, Any]:
    """
    Load performance data from CSV file.

    Parameters:
    -----------
    metric_config : dict
        Configuration for the metric (name, units, higher_is_better)

    Returns:
    --------
    dict : Contains metric_config and data
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
        raw_data = pd.to_numeric(df.iloc[:, 0], downcast="integer").to_numpy()

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

    # Set random seed for reproducibility
    random_state = metric_config.get("random_state", 0)

    # Get metric id, Keep only first suffix (denoted by underscore)
    metric_id = "_".join(metric_config["id"].split("_")[:2])

    # Generate synthetic data
    raw_data = generate_synthetic_data(metric_id, n_samples, random_state)

    return {
        "metric_config": metric_config,
        "load": {
            "data": raw_data,
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
    """
    Docstring for load_data.

    Parameters:
    -----------
    metric_config : dict
        Configuration for the metric

    Returns:
    --------
    dict : Contains raw_data, metric_config, and metadata
    """
    # Determine source type and load data accordingly
    source_type = metric_config.get("source_type")

    if source_type not in ["csv", "synthetic"]:
        raise NotImplementedError(f"---> Unknown source_type {source_type} in metric configuration.")

    return (
        _load_from_csv(metric_config) if source_type == "csv"
        else _load_from_synthetic(metric_config)
    )
