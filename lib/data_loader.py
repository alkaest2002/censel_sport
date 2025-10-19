from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.utils import generate_synthetic_data


def load_from_csv(
    filepath: Path,
    column_name: str,
    metric_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Load performance data from CSV file.

    Parameters:
    -----------
    filepath : Path
        Path to CSV file
    column_name : str
        Name of column containing performance data
    metric_config : dict
        Configuration for the metric (name, units, higher_is_better)

    Returns:
    --------
    dict : Contains raw_data, metric_config, and metadata

    Throws:
    --------
        - FileNotFoundError
        - KeyError
    """
    # If file does not exist
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath.name} not found")
    try:
        # Read csv file
        df = pd.read_csv(filepath)
        # Get raw data
        raw_data = pd.to_numeric(df[column_name], downcast="integer").to_numpy()
    except Exception as e:  # noqa: BLE001
        print(e)
        return {
            "raw_data": None,
            "metric_config": metric_config,
            "metadata": {
                "source": f"csv:{filepath.name}:{column_name}",
                "original_size": 0,
                "valid_records": 0,
                "error": str(e),
            },
        }
    else:
        return {
            "raw_data": raw_data,
            "metric_config": metric_config,
            "metadata": {
                "source": f"csv:{filepath.name}:{column_name}",
                "original_size": len(df),
                "valid_records": len(raw_data),
            },
        }


def load_from_array(data: Iterable[float], metric_config: dict[str, Any]) -> dict[str, Any]:
    """
    Load performance data from numpy array or list.

    Parameters:
    -----------
    data : array-like
        Performance data
    metric_config : dict
        Configuration for the metric

    Returns:
    --------
    dict : Contains raw_data, metric_config, and metadata
    """
    # Ensure raw_data is a numpy array
    raw_data = np.array(data)

    return {
        "raw_data": raw_data,
        "metric_config": metric_config,
        "metadata": {
            "source": "array",
            "original_size": len(raw_data),
            "valid_records": len(raw_data),
        },
    }


def load_from_synthetic(
    metric_config: dict[str, Any],
    n_samples: int = 500,
) -> dict[str, Any]:
    """
    Generate synthetic data for testing.

    Parameters:
    -----------
    metric_type : str
        Type of metric (e.g., '25m_swim', 'push_ups')
    metric_config : dict
        Configuration for the metric
    n_samples : int
        Number of samples to generate

    Returns:
    --------
    dict : Contains raw_data, metric_config, and metadata
    """
    raw_data = generate_synthetic_data(metric_config["name"], n_samples)

    return {
        "raw_data": raw_data,
        "metric_config": metric_config,
        "metadata": {
            "source": "synthetic",
            "original_size": len(raw_data),
            "valid_records": len(raw_data),
        },
    }
