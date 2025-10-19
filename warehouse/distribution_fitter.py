from typing import Any

import numpy as np
from scipy import stats


class DistributionFitter:
    """Fit multiple distributions to the data."""

    @staticmethod
    def fit_distributions(data_dict: dict[str, Any], distributions: dict[str, tuple]) -> dict[str, Any]:
        """
        Fit multiple probability distributions to the analysis data.

        Parameters:
        -----------
        data_dict : dict
            Data dictionary with cleaned data

        Returns:
        --------
        dict : Updated data dictionary with fitted models
        """
        fitted_models: dict[str, Any] = {}
        fit_results: dict[str, Any] = {}
        analysis_data: np.ndarray = data_dict["analysis_data"]

        for distribution_name, distribution in distributions.items():
            try:
                distribution_class, distribution_fit_function = distribution
                params = distribution_fit_function(analysis_data)
                distribution_object = distribution_class(*params)

                # Calculate goodness of fit metrics
                if data_dict["metric_type"] == "count":
                    # For count distributions
                    unique_vals = np.unique(analysis_data)
                    observed_freq = np.array([np.mean(analysis_data == val) for val in unique_vals])
                    expected_freq = distribution_object.pmf(unique_vals)
                    ks_stat = np.max(np.abs(np.cumsum(observed_freq) - np.cumsum(expected_freq)))
                    ks_pvalue = None  # Not easily calculated for custom discrete distributions
                    log_likelihood = np.sum(distribution_object.logpdf(analysis_data))
                else:
                    # For time distributions
                    ks_stat, ks_pvalue = stats.kstest(analysis_data, distribution_object.cdf)
                    log_likelihood = np.sum(distribution_object.logpdf(analysis_data))

                k = len(params)
                n = len(analysis_data)
                aic = 2*k - 2*log_likelihood
                bic = k*np.log(n) - 2*log_likelihood

                fitted_models[distribution_name] = {
                    "distribution": distribution_object,
                    "parameters": params,
                }

                fit_results[distribution_name] = {
                    "ks_statistic": ks_stat,
                    "ks_pvalue": ks_pvalue,
                    "log_likelihood": log_likelihood,
                    "aic": aic,
                    "bic": bic,
                    "n_parameters": k,
                }

            except Exception as e:  # noqa: BLE001
                print(f"Warning: Could not fit {distribution_name} distribution: {e}")
                continue

        if not fitted_models:
            raise ValueError("No distributions could be fitted successfully!")

        data_dict.update({
            "fitted_models": fitted_models,
            "fit_results": fit_results,
        })

        return data_dict
