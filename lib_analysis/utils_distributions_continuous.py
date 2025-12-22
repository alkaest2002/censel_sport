from scipy import stats


def get_continuous_distributions() -> dict[str, type[stats.rv_continuous | stats.rv_discrete]]:
    """Get a mapping of distribution names to their corresponding SciPy distribution objects.

    Returns:
        dict: Mapping of distribution names to distribution classes.
    """
    return {
        "skew_normal": stats.skewnorm,
        "log_normal": stats.lognorm,
        "gamma": stats.gamma,
        "weibull": stats.weibull_min,
    }
