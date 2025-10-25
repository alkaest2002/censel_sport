import sys
from typing import Any

from lib.data_bootstrap import compute_bootstrap_percentiles
from lib.data_clean import clean_data
from lib.data_fit import DistributionFitter
from lib.data_load import load_data
from lib.data_montecarlo import monte_carlo_validation
from lib.data_plot import create_plots
from lib.data_save import save_analysis_results
from lib.data_standardize import compute_standard_scores
from lib.utils_generic import load_configuration_data, parse_arguments, validate_file_path


def main() -> int:
    """
    Run the full data analysis pipeline.

    Returns:
    --------
    int
        Process exit status code:
        - 0: Success
        - 1: Error in file validation or loading
    """
    # Parse command line arguments
    args = parse_arguments()

    try:
        # Validate the file path
        validated_path = validate_file_path(args.filepath, "analysis")

    # Except block to catch file validation errors
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1

    try:
        # Load metric configuration
        metric_config: dict[str, Any] = load_configuration_data(validated_path)

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

    # Catch-all for unexpected errors
    except Exception as e:  # noqa: BLE001
        print(e)
        return 1

    # If everything went well
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
