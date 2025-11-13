from scipy import stats


def get_continuous_distributions() -> dict[str, stats.rv_continuous | stats.rv_discrete]:
    """Get a mapping of distribution names to their corresponding SciPy distribution objects.

    Returns:
        dict[str, stats.rv_continuous | stats.rv_discrete]: A dictionary mapping
            distribution names (str) to their corresponding SciPy distribution
            class objects. Currently includes exponential, skew normal, log normal,
            gamma, and Weibull distributions.
    """
    return {
        "exponential": stats.expon,
        "skew_normal": stats.skewnorm,
        "log_normal": stats.lognorm,
        "gamma": stats.gamma,
        "weibull": stats.weibull_min,
    }
