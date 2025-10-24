# mypy: disable-error-code="misc"

from collections.abc import Hashable
from itertools import count
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from . import MT100, MT1000, PUSHUPS, SITUPS, SWIM25


def generate_synthetic_data(
        metric_type: str,
        n_samples: int = 500,
        random_state: int = 42) -> Any:
    """
    Generate synthetic performance data for testing purposes.

    Parameters:
    -----------
    metric_type : str
        Type of metric to generate data for

    n_samples : int
        Number of samples to generate

    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    np.array : Generated performance data
    """

    rng = np.random.default_rng(random_state)

    base_synthetic_data = {
        SWIM25: {
            "distributions": [
                (rng.normal, (11.5, 0.8), 0.05),
                (rng.normal, (14.0, 1.0), 0.15),
                (rng.normal, (18.5, 2.5), 0.60),
                (rng.normal, (25.0, 3.0), 0.20),
            ],
            "bounds": (8.0, 40.0),
            "discrete": False,
            "unit": "seconds",
        },
        MT100: {
            "distributions": [
                (rng.normal, (10.0, 0.3), 0.05),
                (rng.normal, (11.5, 0.5), 0.15),
                (rng.normal, (13.5, 1.0), 0.60),
                (rng.normal, (16.5, 1.5), 0.20),
            ],
            "bounds": (8.0, 25.0),
            "discrete": False,
            "unit": "seconds",
        },
        MT1000: {
            "distributions": [
                (rng.normal, (150, 12), 0.05),  # ~2.5 minutes
                (rng.normal, (192, 18), 0.15),  # ~3.2 minutes
                (rng.normal, (270, 48), 0.60),  # ~4.5 minutes
                (rng.normal, (390, 60), 0.20),  # ~6.5 minutes
            ],
            "bounds": (120, 600),  # 2-10 minutes
            "discrete": False,
        },
        SITUPS: {
            "distributions": [
                (rng.normal, (60, 8), 0.05),
                (rng.normal, (45, 6), 0.15),
                (rng.normal, (30, 8), 0.60),
                (rng.normal, (15, 5), 0.20),
            ],
            "bounds": (0, None),
            "discrete": True,
            "unit": "counts",
        },
        PUSHUPS: {
            "distributions": [
                (rng.normal, (75, 10), 0.05),
                (rng.normal, (55, 8), 0.15),
                (rng.normal, (30, 10), 0.60),
                (rng.normal, (12, 4), 0.20),
            ],
            "bounds": (0, None),
            "discrete": True,
            "unit": "counts",
        },
    }

    # Check if metric type is known
    if metric_type not in base_synthetic_data:
        raise ValueError(f"---> Unknown metric_type {metric_type} for synthetic data generation.")

    # Get configuration for the specified metric type
    config: dict[str, Any] = base_synthetic_data[metric_type]

    # Generate data from multiple distributions
    data_parts = []
    for func, params, proportion in config["distributions"]:
        size = int(n_samples * proportion)
        data_parts.append(func(*params, size))

    # Concatenate data
    data = np.concatenate(data_parts)

    # Apply bounds
    lower, upper = config["bounds"]
    if lower is not None:
        data = data[data > lower]
    if upper is not None:
        data = data[data < upper]

    # Make discrete if needed
    if config.get("discrete", False):
        data = np.round(data).astype(int)

    return data


def apply_standardization(
        data_to_standardize: NDArray[np.integer[Any] | np.floating[Any]],
        cutoffs: list[tuple],
        higher_is_better: bool = False,
    ) -> list[dict[Hashable, Any]]:
    """
    Standardize data with percentile cutoffs.

    Parameters:
    -----------
    data_to_standardize : np.array
        Data to be standardized

    cutoffs : list of tuples
        List of tuples containing percentile cutoffs

    higher_is_better : bool
        Whether higher values indicate better performance

    Returns:
    --------
    list : Original data and Standardized data
    """

    # Convert data to pandas Series for easier manipulation
    data = pd.Series(data_to_standardize)

    # Add inclusive bounds to cutoffs
    # First cutoff is inclusive on both sides, others only on the right
    cutoffs_with_inclusive = list(zip(cutoffs, ["both", *["right"] * (len(cutoffs) - 1)], strict=True))

    # initialize counter based on whether higher values denotes better performance
    counter = count(start=1, step=1) if higher_is_better else count(start=len(cutoffs), step=-1)

    # Compute standardized scores
    standardized_scores = data.case_when(
        [
            (lambda x, cutoffs=cutoffs, inclusive=inclusive:
                x.between(cutoffs[0], cutoffs[1], inclusive=inclusive), next(counter))
            for (cutoffs, inclusive) in cutoffs_with_inclusive
        ],
    )

    # Compute standardized scores lower bounds for data
    standardized_bounds = data.case_when(
        [
            (lambda x, cutoffs=cutoffs, inclusive=inclusive:
                x.between(cutoffs[0], cutoffs[1], inclusive=inclusive), f"{cutoffs[0]} - {cutoffs[1]}")
            for (cutoffs, inclusive) in cutoffs_with_inclusive
        ],
    )

    return pd.concat(
        [
            data,
            standardized_scores,
            standardized_bounds,
        ],
        keys=["original_value","standardized_value", "standardized_value_bounds"],
        axis=1,
    ).to_dict(orient="records")

def _generate_random_samples_from_model(  # noqa: PLR0911
    best_model: dict[str, Any],
    size: int,
    rng: np.random.Generator,
) -> NDArray[np.floating[Any]]:
    """
    Generate random samples from the best fitted model using numpy random generators.

    Parameters:
    -----------
    best_model : dict
        Dictionary containing model name and parameters
    size : int
        Number of samples to generate
    rng : np.random.Generator
        Random number generator instance

    Returns:
    --------
    NDArray
        Array of random samples from the fitted distribution
    """
    # Extract model name and parameters
    model_name: str = best_model.get("name", "")
    parameters: list[float] = best_model.get("parameters", [])

    if not model_name or not parameters:
        raise ValueError("Invalid best_model dictionary: missing name or parameters")

    # Map distribution names to numpy random generators
    if model_name == "normal":
        loc, scale = parameters
        return rng.normal(loc=loc, scale=scale, size=size)

    if model_name == "lognormal":
        s, loc, scale = parameters
        # scipy.stats lognormal uses (s, loc, scale) but numpy uses (mean, sigma)
        # Convert: mean = log(scale), sigma = s
        return rng.lognormal(mean=np.log(scale), sigma=s, size=size) + loc

    if model_name == "gamma":
        a, loc, scale = parameters
        return rng.gamma(shape=a, scale=scale, size=size) + loc

    if model_name in {"weibull", "Weibull"}:
        c, loc, scale = parameters
        return rng.weibull(a=c, size=size) * scale + loc

    if model_name == "negative_binomial":
        n, p = parameters
        return rng.negative_binomial(n=n, p=p, size=size).astype(float)

    if model_name == "poisson":
        lam = parameters[0]
        return rng.poisson(lam=lam, size=size).astype(float)

    if model_name == "zero_inflated_poisson":
        # Parameters: [zero_prob, lambda]
        zero_prob, lam = parameters

        # Generate from mixture: zeros with prob zero_prob, Poisson otherwise
        mask = rng.random(size=size) < zero_prob
        samples = rng.poisson(lam=lam, size=size).astype(float)
        samples[mask] = 0.0
        return samples

    # Fallback to generic random (uniform distribution)
    print(f"---> Warning: Unknown distribution '{model_name}', falling back to uniform random")
    return rng.random(size=size)

