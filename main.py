
from pathlib import Path
from typing import Any

import orjson

from lib.data_bootstrap import compute_bootstrap_percentiles
from lib.data_clean import clean_data
from lib.data_fit import DistributionFitter
from lib.data_load import load_data
from lib.data_montecarlo import monte_carlo_validation
from lib.data_plot import create_plots
from lib.data_save import save_analysis_results
from lib.data_standardize import compute_standard_scores

try:

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
        step_counter = 1
        print(f"{step_counter}. Loading data for {metric_config.get('title')}...")
        data_dict: dict[str, Any] = load_data(metric_config=metric_config)

        ############################################################################################
        # Clean data
        ############################################################################################
        step_counter += 1
        print(f"{step_counter}. Cleaning data...")
        data_dict = clean_data(data_dict)

        ###############################################################################################
        # Fit theoretical distributions to data
        ###############################################################################################
        step_counter += 1
        print(f"{step_counter}. Fitting theoretical distributions...")
        fitter = DistributionFitter(data_dict)
        data_dict = fitter.fit_distributions()


        #############################################################################################
        # Compute bootstrap percentiles
        #############################################################################################
        step_counter += 1
        print(f"{step_counter}. Computing bootstrap percentiles...")
        data_dict, bootstrap_samples = compute_bootstrap_percentiles(data_dict=data_dict)

        ##############################################################################################
        # Apply percentile-based standardization to data
        ##############################################################################################
        step_counter += 1
        print(f"{step_counter}. Applying standardization...")
        data_dict = compute_standard_scores(data_dict)

        ###############################################################################################
        # Perform Montecarlo simulation
        ###############################################################################################
        step_counter += 1
        print(f"{step_counter}. Performing Montecarlo simulation...")
        data_dict, simulation_samples = monte_carlo_validation(data_dict)

        ###############################################################################################
        # Create plots
        ###############################################################################################
        step_counter += 1
        print(f"{step_counter}. Saving plots...")
        data_dict = create_plots(data_dict)

        ##############################################################################################
        # Save results
        ##############################################################################################
        step_counter += 1
        print(f"{step_counter}. Saving results...")
        save_analysis_results(
            data_dict=data_dict,
            bootstrap_samples=bootstrap_samples,
            simulation_samples=simulation_samples,
        )
except Exception as e:  # noqa: BLE001
    print(e)
