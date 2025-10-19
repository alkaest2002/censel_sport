from typing import Any

import numpy as np

# To change percentile cutoffs globally, modify the constant at the top
# PERCENTILE_CUTOFFS = [1, 5, 20, 80, 95, 99]  # More extreme
# PERCENTILE_CUTOFFS = [5, 15, 30, 70, 85, 95]  # More conservative
PERCENTILE_CUTOFFS = [2.5, 10, 25, 75, 90, 97.5]

class PercentileCalculator:
    """Step 5: Calculate relevant percentiles from the best distribution."""

    @staticmethod
    def calculate_percentiles(data_dict: dict[str, Any], percentiles:list[float]|None=None) -> dict[str, Any]:
        """
        Calculate percentiles from the best fitting distribution.

        Parameters:
        -----------
        data_dict : dict
            Data dictionary with best model
        percentiles : list
            List of percentiles to calculate

        Returns:
        --------
        dict : Updated data dictionary with calculated percentiles
        """
        if percentiles is None:
            percentiles = PERCENTILE_CUTOFFS

        best_model = data_dict["best_model"]["distribution"]
        higher_is_better = data_dict["metric_config"]["higher_is_better"]

        # Calculate percentiles in analysis scale
        percentile_values_analysis = [best_model.ppf(p/100) for p in percentiles]

        # Convert back to original scale if needed
        if higher_is_better:
            max_val = np.max(data_dict["clean_data"])
            percentile_values_original = [max_val + 1 - val for val in percentile_values_analysis]
        else:
            percentile_values_original = percentile_values_analysis

        data_dict.update({
            "percentiles": percentiles,
            "percentile_values_analysis": percentile_values_analysis,
            "percentile_values_original": percentile_values_original,
        })

        return data_dict
