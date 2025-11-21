from base64 import b64decode
from pathlib import Path
from typing import Any

import orjson

from lib_analysis.utils_generic import is_falsy

data_out = Path("./data_out")

def save_analysis_results(
    data_dict: dict[str, Any],
    bootstrap_samples: list[Any],
    simulation_samples: list[Any],
) -> None:
    """Save analysis results and bootstrap estimates to JSON files.

    This function extracts analysis data from a dictionary and saves it to multiple
    files including SVG plots, analysis results, bootstrap samples, and Monte Carlo
    simulation samples. All files are saved to a directory named after the metric ID.

    Args:
        data_dict: Dictionary containing data.
        bootstrap_samples: List containing bootstrap samples for statistical analysis.
        simulation_samples: List containing Monte Carlo simulation samples.

    Returns:
        None

    Raises:
        ValueError: If data_dict is missing required keys ('metric_config', 'plots')
            or if metric_config is missing the 'id' key.
    """
    # Extract data from dictionary
    metric_config: dict[str, Any] = data_dict.get("metric_config", {})
    plots: list[dict[str, str]] = data_dict.get("plots", {})
    metric_id: str = metric_config.get("id", "")

    # Raise error if something crucial is missing
    if any(map(is_falsy, (metric_config, plots, metric_id))):
        raise ValueError("---> The data dictionary does not contain all required parts.")

    # Determine output folder
    output_path: Path = data_out / metric_id

    # Make sure output folder exists or create it
    output_path.mkdir(parents=True, exist_ok=True)

    # Delete existing files in folder if exists
    for child in output_path.iterdir():
        if child.is_file():
            child.unlink()

    # Write plots to SVG files
    for plot in plots:

        # Create output path for plot
        plot_output_path: Path = output_path / f"{metric_id}_{plot['name']}.svg"

        # Write SVG plot to file
        with plot_output_path.open("w") as f:
            # Strip 'data:image/svg+xml;base64,'
            plot_str: str = plot["svg"][26:]
            # Write svg to file
            f.write(b64decode(plot_str).decode("utf-8"))

    # Convert query_from_db to a serializable format
    if "query_from_db" in data_dict:
        query_df = data_dict["query_from_db"]
        if query_df is not None:
            data_dict["query_from_db"] = query_df.to_dict(orient="records")
        else:
            data_dict["query_from_db"] = None

    # Write data analysis to JSON file
    analysis_output_path: Path = output_path / f"{metric_id}_analysis.json"
    with analysis_output_path.open("w") as f:
        orjson_options: int = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2
        f.write(orjson.dumps(data_dict, option=orjson_options).decode("utf-8"))

    # Write bootstrap samples to JSON file
    bootstrap_output_path: Path = output_path / f"{metric_id}_bootstrap_samples.json"
    with bootstrap_output_path.open("w") as f:
        orjson_options = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2
        f.write(orjson.dumps(bootstrap_samples, option=orjson_options).decode("utf-8"))

    # Write Monte Carlo samples to JSON file
    montecarlo_output_path: Path = output_path / f"{metric_id}_montecarlo_samples.json"
    with montecarlo_output_path.open("w") as f:
        orjson_options = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2
        f.write(orjson.dumps(simulation_samples, option=orjson_options).decode("utf-8"))
