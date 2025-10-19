

from itertools import pairwise
from typing import Any

import numpy as np


def compute_percentile_cutoffs(
        bootstrap_percentiles: dict[str, Any],
        precision: int = 2,
    ) -> list[tuple[float, float]]:
    """
    Create a normative table using bootstrap percentiles.

    Parameters:
    -----------
    bootstrap_percentiles : dict
        Data dictionary with bootstrap percentile data
    precision : int
        Decimal precision for rounding cutoffs

    Returns:
        list: A list containing the normative table cutoffs.
    """
    # Extract percentile values
    percentiles_values = [percentile["value"] for percentile in bootstrap_percentiles]

    # Create normative table cutoffs
    corrected_percentiles_values = np.round([0, *percentiles_values, 1e10], precision)

    return list(pairwise(corrected_percentiles_values))
