from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils import resample

from lib.percentiles_bootstrap import compute_bootstrap_percentiles


class NormativeTableValidator:
    """Step 7: Validate normative table using bootstrap and Monte Carlo."""

    @staticmethod
    def bootstrap_validation(data_dict: dict[str, Any], n_bootstrap: int=5000) -> dict[str, Any]:
        """
        Validate normative table using bootstrap resampling of observed data.

        Parameters:
        -----------
        data_dict : dict
            Data dictionary with normative table
        n_bootstrap : int
            Number of bootstrap samples

        Returns:
        --------
        dict : Updated data dictionary with bootstrap results
        """
        clean_data = data_dict["clean_data"]
        normative_table = data_dict["normative_table"]
        higher_is_better = data_dict["metric_config"]["higher_is_better"]

        bootstrap_results = []

        print(f"Running {n_bootstrap} bootstrap validations...")

        for i in range(n_bootstrap):
            if (i + 1) % 200 == 0:
                print(f"Completed {i + 1}/{n_bootstrap} bootstrap samples")

            # Resample data
            bootstrap_sample = resample(clean_data, n_samples=len(clean_data), random_state=i)

            # Calculate level distribution for bootstrap sample
            level_counts = [0] * 7
            for value in bootstrap_sample:
                level = NormativeTableValidator.classify_single_value(
                    value, normative_table, higher_is_better,
                )
                if level is not None:
                    level_counts[level] += 1

            level_percentages = [(count / len(bootstrap_sample)) * 100 for count in level_counts]
            bootstrap_results.append(level_percentages)

        data_dict["bootstrap_results"] = np.array(bootstrap_results)
        return data_dict

    @staticmethod
    def montecarlo_validation(data_dict: dict[str, Any], n_simulations: int=10000) -> dict[str, Any]:
        """
        Validate normative table using Monte Carlo simulation from fitted distribution.

        Parameters:
        -----------
        data_dict : dict
            Data dictionary with best model
        n_simulations : int
            Number of Monte Carlo simulations

        Returns:
        --------
        dict : Updated data dictionary with Monte Carlo results
        """
        best_model = data_dict["best_model"]["distribution"]
        normative_table = data_dict["normative_table"]
        higher_is_better = data_dict["metric_config"]["higher_is_better"]
        n_samples = len(data_dict["clean_data"])

        montecarlo_results = []

        print(f"Running {n_simulations} Monte Carlo simulations...")

        for i in range(n_simulations):
            if (i + 1) % 200 == 0:
                print(f"Completed {i + 1}/{n_simulations} simulations")

            # Generate data from fitted distribution (analysis scale)
            simulated_analysis = best_model.rvs(size=n_samples, random_state=i)

            # Convert to original scale
            if higher_is_better:
                max_val = np.max(data_dict["clean_data"])
                simulated_original = max_val + 1 - simulated_analysis
            else:
                simulated_original = simulated_analysis

            # Calculate level distribution
            level_counts = [0] * 7
            for value in simulated_original:
                level = NormativeTableValidator.classify_single_value(
                    value, normative_table, higher_is_better,
                )
                if level is not None:
                    level_counts[level] += 1

            level_percentages = [(count / len(simulated_original)) * 100 for count in level_counts]
            montecarlo_results.append(level_percentages)

        data_dict["montecarlo_results"] = np.array(montecarlo_results)
        return data_dict

    @staticmethod
    def classify_single_value(value: float, normative_table: pd.DataFrame, higher_is_better: bool) -> int | None:
        """Helper function to classify a single value."""
        for level in range(7):
            mask = NormativeTableCreator.classify_data_mask(
                np.array([value]), level, normative_table, higher_is_better,
            )
            if mask[0]:
                return level
        return None
