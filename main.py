
from pathlib import Path
from typing import Any

import orjson

from lib.data_cleaner import clean_data
from lib.data_loader import load_from_csv, load_from_synthetic
from lib.data_saver import save_analysis_results
from lib.percentiles_bootstrap import compute_bootstrap_percentiles
from lib.percentiles_cutoffs import compute_percentile_cutoffs

# Iterate over all metric configuration files in data_in folder
for metric_config_path in Path("./data_in").glob("*.json"):


    # Open metric configuration file
    with metric_config_path.open("r") as f:

        # Parse metric configuration
        metric_config: dict[str, Any] = orjson.loads(f.read())

    #################################################################################
    # Load data
    ################################################################################
    print("1. Loading data...")

    # Get source type and load data accordingly
    source_type = metric_config.get("source_type")

    # Load csv data
    if source_type == "csv":
        data_dict: dict[str, Any] = load_from_csv(metric_config=metric_config)
    # Load synthetic data
    elif source_type == "synthetic":
        data_dict: dict[str, Any] = load_from_synthetic(metric_config=metric_config)
    # Unknown source type
    else:
        raise NotImplementedError(f"Unknown source_type {source_type} in metric configuration.")

    #################################################################################
    # Clean data
    #################################################################################
    print("2. Cleaning data...")

    # Clean data
    data_dict = clean_data(data_dict)


    #################################################################################
    # Compute bootstrap percentiles
    #################################################################################
    print("3. Computing bootstrap percentiles...")

    # Compute bootstrap percentiles
    bootstrap_percentiles, bootstrap_estimates = (
        compute_bootstrap_percentiles(
            data_dict["analysis_data"],
            requested_percentiles=metric_config["requested_percentiles"],
            n_replicates=metric_config["bootstrap_n_replicates"],
            n_replicate_size=metric_config["bootstrap_n_replicate_size"],
        )
    )


    #################################################################################
    # Create normative table
    #################################################################################
    print("4. Creating normative table...")

    # Create normative table
    percentile_cutoffs = compute_percentile_cutoffs(bootstrap_percentiles, metric_config.get("precision", 2))

    # Store results in data dictionary
    data_dict["normative_table"] = {
        "requested_percentiles": metric_config["requested_percentiles"],
        "bootstrap_percentiles": bootstrap_percentiles,
        "computed_cutoffs": percentile_cutoffs,
    }


    #################################################################################
    # Save results
    #################################################################################
    print("5. Saving results...")

    save_analysis_results(
        metric_config=metric_config,
        data_dict=data_dict,
        bootstrap_estimates=bootstrap_estimates,
    )
