
from pathlib import Path
from typing import Any

import orjson

data_out = Path("./data_out")

def save_analysis_results(
    data_dict: dict[str, Any],
    bootstrap_samples: list[Any]) -> None:
    """
    Save analysis results and bootstrap estimates to JSON files.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing data

    bootstrap_samples : dict
        Dictionary containing bootstrap estimates

    Returns:
    --------
    None
    """

    # Extract metric name from data dictionary
    metric_config: dict[str, Any] = data_dict.get("metric_config", {})
    metric_name = metric_config.get("name", "unknown_metric")

    # Determine output folder
    output_path = data_out / metric_name

    # Make sure output folder exists or create it
    output_path.mkdir(parents=True, exist_ok=True)

    # Delete existing files in folder if exists
    for child in output_path.iterdir():
        if child.is_file():
            child.unlink()

    # Write results to JSON file
    analysis_output_path = output_path / f"{metric_name}_analysis.json"
    with analysis_output_path.open("w") as f:
        orjson_options = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2
        f.write(orjson.dumps(data_dict, option=orjson_options).decode("utf-8"))

    # Write bootstrap estimates to separate JSON file
    bootstrap_output_path = output_path / f"{metric_name}_bootstrap_samples.json"
    with bootstrap_output_path.open("w") as f:
        orjson_options = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2
        f.write(orjson.dumps(bootstrap_samples, option=orjson_options).decode("utf-8"))
