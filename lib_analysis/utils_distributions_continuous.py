
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FitFunctionType = Callable[[NDArray[np.integer[Any] | np.floating[Any]]], tuple[float, ...]]

def get_continuous_distributions(
    ) -> dict[str, tuple[stats.rv_continuous | stats.rv_discrete, FitFunctionType]]:
    """
    Fit Poisson distribution parameters to data.

    Returns:
    --------
    dict: dictionary mapping distribution names to (distribution class, fit function) tuples
    """
    # Mapping of distributions
    return {
        "exponential": (stats.expon, lambda x: stats.expon.fit(x)),
        "skew_normal": (stats.skewnorm, lambda x: stats.skewnorm.fit(x)),
        "log_normal": (stats.lognorm, lambda x: stats.lognorm.fit(x, floc=0)),
        "gamma": (stats.gamma, lambda x: stats.gamma.fit(x, floc=0)),
        "weibull": (stats.weibull_min, lambda x: stats.weibull_min.fit(x, floc=0)),
    }
