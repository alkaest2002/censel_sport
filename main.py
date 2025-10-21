
from pathlib import Path
from typing import Any

import orjson

from lib.bootstrap import compute_bootstrap_percentiles
from lib.data_cleaner import clean_data
from lib.data_loader import load_data
from lib.data_standardizer import compute_standard_scores
from lib.data_writer import save_analysis_results
from lib.distribution_fitter import DistributionFitter

# Iterate over all metric configuration files in data_in folder
for metric_config_path in Path("./data_in").glob("*.json"):

    # Open metric configuration file
    with metric_config_path.open("r") as f:

        # Parse metric configuration
        metric_config: dict[str, Any] = orjson.loads(f.read())

        # Skip if included in analysis is set to False
        if metric_config.get("include_in_analysis", False) is False:
            continue

    #################################################################################
    # Load data
    ################################################################################
    print("1. Loading data...")
    data_dict: dict[str, Any] = load_data(metric_config=metric_config)

    ############################################################################################
    # Clean data
    ############################################################################################
    print("2. Cleaning data...")
    data_dict = clean_data(data_dict)

    #############################################################################################
    # Compute bootstrap percentiles
    #############################################################################################
    print("3. Computing bootstrap percentiles...")
    data_dict, bootstrap_samples = compute_bootstrap_percentiles(data_dict=data_dict)

    ##############################################################################################
    # Apply standardization to data
    ##############################################################################################
    print("4. Applying standardization...")
    data_dict = compute_standard_scores(data_dict)

    ###############################################################################################
    # Fit theoretical distributions to data
    ###############################################################################################
    print("5. Fitting theoretical distributions...")
    fitter = DistributionFitter(data_dict)
    data_dict = fitter.fit_distributions()

    ###############################################################################################
    # Montecarlo simulation
    ###############################################################################################
    print("6. Performing Montecarlo simulation...")

    ###############################################################################################
    # Save plots
    ###############################################################################################
    print("7. Saving plots...")


    ##############################################################################################
    # Save results
    ##############################################################################################
    print("8. Saving results...")
    save_analysis_results(
        data_dict=data_dict,
        bootstrap_samples=bootstrap_samples,
    )
