
from base64 import b64decode
from pathlib import Path
from typing import Any

import orjson

from lib.utils_generic import is_falsy

data_out = Path("./data_out")

def save_analysis_results(
    data_dict: dict[str, Any],
    bootstrap_samples: list[Any],
    simulation_samples: list[Any],
    ) -> None:
    """
    Save analysis results and bootstrap estimates to JSON files.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing data

    bootstrap_samples : list
        list containing bootstrap samples

    simulation_samples: list
        list containing montecarlo samples

    Returns:
    --------
    None
    """

    # Extract data from dictionary
    metric_config: dict[str, Any] = data_dict.get("metric_config", {})
    plots: list[dict[str, str]] = data_dict.get("plots", {})
    metric_id: str = metric_config.get("id", "")

    # Raise error if something is missing
    if any(map(is_falsy, (metric_config, plots, metric_id))):
        raise ValueError("---> The data dictionary does not contain all required parts.")

    # Determine output folder
    output_path = data_out / metric_id

    # Make sure output folder exists or create it
    output_path.mkdir(parents=True, exist_ok=True)

    # Delete existing files in folder if exists
    for child in output_path.iterdir():
        if child.is_file():
            child.unlink()

    # Write plots to SVG files
    for plot in plots:
        plot_output_path = output_path / f"{metric_id}_{plot['name']}.svg"
        with plot_output_path.open("w") as f:
            plot_str: str = plot["svg"][26:]
            f.write(b64decode(plot_str).decode("utf-8"))

    # Write results to JSON file
    analysis_output_path = output_path / f"{metric_id}_analysis.json"
    with analysis_output_path.open("w") as f:
        orjson_options = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2
        f.write(orjson.dumps(data_dict, option=orjson_options).decode("utf-8"))

    # Write bootstrap samples to JSON file
    bootstrap_output_path = output_path / f"{metric_id}_bootstrap_samples.json"
    with bootstrap_output_path.open("w") as f:
        orjson_options = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2
        f.write(orjson.dumps(bootstrap_samples, option=orjson_options).decode("utf-8"))

    # Write montecarlo samples to JSON file
    mtntecarlo_output_path = output_path / f"{metric_id}_montecarlo_samples.json"
    with mtntecarlo_output_path.open("w") as f:
        orjson_options = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2
        f.write(orjson.dumps(simulation_samples, option=orjson_options).decode("utf-8"))
