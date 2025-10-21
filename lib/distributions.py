from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy import stats
import statsmodels.api as sm

if TYPE_CHECKING:
    from statsmodels.discrete.discrete_model import DiscreteResults

# ruff: noqa: BLE001
# ruff: noqa: SLF001


class StatsModelsPoissonDist:
    """Poisson distribution using statsmodels."""

    def __init__(self) -> None:
        self.fitted_model: DiscreteResults | None = None
        self._lambda: float | None = None

    @classmethod
    def create(cls, lambda_: float) -> "StatsModelsPoissonDist":
        """Create a StatsModelsPoissonDist instance with given lambda parameter."""
        instance = cls()
        instance._lambda = lambda_
        return instance

    @classmethod
    def fit(cls, data: NDArray[np.integer[Any] | np.floating[Any]]) -> "StatsModelsPoissonDist":
        """
        Fit Poisson distribution to data using statsmodels.

        Parameters:
        -----------
        data : NDArray[np.integer[Any] | np.floating[Any]]
            Data to fit

        Returns:
        --------
        StatsModelsPoissonDist : Fitted distribution instance
        """
        instance = cls()

        # For Poisson regression with only intercept (constant term)
        # This estimates the mean (lambda) parameter
        endog = np.asarray(data, dtype=int)
        exog = np.ones(len(endog))  # Only intercept

        try:
            model = sm.Poisson(endog, exog)
            instance.fitted_model = model.fit(disp=False, maxiter=1000)
            # Extract lambda from the fitted intercept
            instance._lambda = float(np.exp(instance.fitted_model.params[0]))
        except Exception:
            # Fall back to simple MLE (mean)
            instance._lambda = float(np.mean(data))

        return instance

    @classmethod
    def fit_parameters(cls, data: NDArray[np.integer[Any] | np.floating[Any]]) -> tuple[float]:
        """
        Fit Poisson distribution parameters to data.

        Parameters:
        -----------
        data : NDArray[np.integer[Any] | np.floating[Any]]
            Data to fit

        Returns:
        --------
        tuple[float] : Tuple containing (lambda,) parameter
        """
        fitted_dist = cls.fit(data)
        if fitted_dist._lambda is None:
            raise ValueError("Failed to fit Poisson distribution parameters")
        return (fitted_dist._lambda,)

    def pmf(self, k: NDArray[np.integer[Any] | np.floating[Any]]) -> NDArray[np.integer[Any] | np.floating[Any]]:
        """Probability mass function."""
        if self._lambda is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return cast("NDArray[np.integer[Any] | np.floating[Any]]", stats.poisson.pmf(k, self._lambda))

    def logpmf(self, k: NDArray[np.integer[Any] | np.floating[Any]]) -> NDArray[np.integer[Any] | np.floating[Any]]:
        """Log probability mass function."""
        if self._lambda is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return cast("NDArray[np.integer[Any] | np.floating[Any]]", stats.poisson.logpmf(k, self._lambda))

    def pdf(self, k: NDArray[np.integer[Any] | np.floating[Any]]) -> NDArray[np.integer[Any] | np.floating[Any]]:
        """PDF method for compatibility (same as PMF for discrete distribution)."""
        return self.pmf(k)

    def logpdf(self, k: NDArray[np.integer[Any] | np.floating[Any]]) -> NDArray[np.integer[Any] | np.floating[Any]]:
        """Log PDF method for compatibility (same as log PMF for discrete distribution)."""
        return self.logpmf(k)

    def cdf(self, k: NDArray[np.integer[Any] | np.floating[Any]]) -> NDArray[np.integer[Any] | np.floating[Any]]:
        """Cumulative distribution function."""
        if self._lambda is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return cast("NDArray[np.integer[Any] | np.floating[Any]]", stats.poisson.cdf(k, self._lambda))

    def rvs(self, size: int | tuple[int, ...] = 1, random_state: int = 42) -> NDArray[np.integer[Any]]:
        """Generate random variates."""
        if self._lambda is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return cast("NDArray[np.integer[Any]]", stats.poisson.rvs(self._lambda, size=size, random_state=random_state))

    def mean(self) -> float:
        """Mean of the distribution."""
        if self._lambda is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return float(self._lambda)

    def var(self) -> float:
        """Variance of the distribution."""
        if self._lambda is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return float(self._lambda)


class StatsModelsNegativeBinomialDist:
    """Negative Binomial distribution using statsmodels."""

    def __init__(self) -> None:
        self.fitted_model: DiscreteResults | None = None
        self.mu: float | None = None
        self.alpha: float | None = None

    @classmethod
    def create(cls, mu: float, alpha: float) -> "StatsModelsNegativeBinomialDist":
        """Create a StatsModelsNegativeBinomialDist instance with given parameters."""
        instance = cls()
        instance.mu = mu
        instance.alpha = alpha
        return instance

    @classmethod
    def fit(cls, data: NDArray[np.integer[Any] | np.floating[Any]]) -> "StatsModelsNegativeBinomialDist":
        """
        Fit Negative Binomial distribution to data using statsmodels.

        Parameters:
        -----------
        data : NDArray[np.integer[Any] | np.floating[Any]]
            Data to fit

        Returns:
        --------
        StatsModelsNegativeBinomialDist : Fitted distribution instance
        """
        instance = cls()

        # For Negative Binomial regression with only intercept
        endog = np.asarray(data, dtype=int)
        exog = np.ones(len(endog))  # Only intercept

        try:
            model = sm.NegativeBinomial(endog, exog)
            instance.fitted_model = model.fit(disp=False, maxiter=1000)
            # Extract parameters
            instance.mu = float(np.exp(instance.fitted_model.params[0]))
            instance.alpha = float(instance.fitted_model.params[1])  # Dispersion parameter
        except Exception:
            # Fall back to method of moments
            mean = float(np.mean(data))
            var = float(np.var(data))

            if var > mean:
                # alpha = (var - mean) / mean^2
                instance.alpha = (var - mean) / (mean**2)
                instance.mu = mean
            else:
                # Fall back to Poisson-like behavior
                instance.alpha = 0.001
                instance.mu = mean

        return instance

    @classmethod
    def fit_parameters(cls, data: NDArray[np.integer[Any] | np.floating[Any]]) -> tuple[float, float]:
        """
        Fit Negative Binomial distribution parameters to data.

        Parameters:
        -----------
        data : NumberArray
            Data to fit

        Returns:
        --------
        tuple[float, float] : Tuple containing (mu, alpha) parameters
        """
        fitted_dist = cls.fit(data)
        if fitted_dist.mu is None or fitted_dist.alpha is None:
            raise ValueError("Failed to fit Negative Binomial distribution parameters")
        return (fitted_dist.mu, fitted_dist.alpha)

    def _convert_to_scipy_params(self) -> tuple[float, float]:
        """Convert statsmodels parameterization to scipy parameterization."""
        if self.mu is None or self.alpha is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")

        # Convert from statsmodels (mu, alpha) to scipy (n, p) parameterization
        # mu = n * (1-p) / p
        # alpha = 1 / n
        n = 1.0 / self.alpha
        p = n / (n + self.mu)
        return n, p

    def pmf(self, k: NDArray[np.integer[Any] | np.floating[Any]]) -> NDArray[np.integer[Any] | np.floating[Any]]:
        """Probability mass function."""
        n, p = self._convert_to_scipy_params()
        return cast("NDArray[np.integer[Any] | np.floating[Any]]", stats.nbinom.pmf(k, n, p))

    def logpmf(self, k: NDArray[np.integer[Any] | np.floating[Any]]) -> NDArray[np.integer[Any] | np.floating[Any]]:
        """Log probability mass function."""
        n, p = self._convert_to_scipy_params()
        return cast("NDArray[np.integer[Any] | np.floating[Any]]", stats.nbinom.logpmf(k, n, p))

    def pdf(self, k: NDArray[np.integer[Any] | np.floating[Any]]) -> NDArray[np.integer[Any] | np.floating[Any]]:
        """PDF method for compatibility (same as PMF for discrete distribution)."""
        return self.pmf(k)

    def logpdf(self, k: NDArray[np.integer[Any] | np.floating[Any]]) -> NDArray[np.integer[Any] | np.floating[Any]]:
        """Log PDF method for compatibility (same as log PMF for discrete distribution)."""
        return self.logpmf(k)

    def cdf(self, k: NDArray[np.integer[Any] | np.floating[Any]]) -> NDArray[np.integer[Any] | np.floating[Any]]:
        """Cumulative distribution function."""
        n, p = self._convert_to_scipy_params()
        return cast("NDArray[np.integer[Any] | np.floating[Any]]", stats.nbinom.cdf(k, n, p))

    def rvs(self, size: int | tuple[int, ...] = 1, random_state: int = 42) -> NDArray[np.integer[Any]]:
        """Generate random variates."""
        n, p = self._convert_to_scipy_params()
        return cast("NDArray[np.integer[Any]]", stats.nbinom.rvs(n, p, size=size, random_state=random_state))

    def mean(self) -> float:
        """Mean of the distribution."""
        if self.mu is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return float(self.mu)

    def var(self) -> float:
        """Variance of the distribution."""
        if self.mu is None or self.alpha is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return float(self.mu + self.alpha * self.mu**2)


class StatsModelsZeroInflatedPoissonDist:
    """Zero-Inflated Poisson distribution using statsmodels."""

    def __init__(self) -> None:
        self.fitted_model: DiscreteResults | None = None
        self.lambda_: float | None = None
        self.pi: float | None = None

    @classmethod
    def create(cls, lambda_: float, pi: float) -> "StatsModelsZeroInflatedPoissonDist":
        """Create a StatsModelsZeroInflatedPoissonDist instance with given parameters."""
        instance = cls()
        instance.lambda_ = lambda_
        instance.pi = pi
        return instance

    @classmethod
    def fit(cls, data: NDArray[np.integer[Any] | np.floating[Any]]) -> "StatsModelsZeroInflatedPoissonDist":
        """
        Fit Zero-Inflated Poisson distribution to data using statsmodels.

        Parameters:
        -----------
        data : NDArray[np.integer[Any] | np.floating[Any]]
            Data to fit

        Returns:
        --------
        StatsModelsZeroInflatedPoissonDist : Fitted distribution instance
        """
        instance = cls()

        # For Zero-Inflated Poisson regression with only intercept
        endog = np.asarray(data, dtype=int)
        exog = np.ones(len(endog))  # Only intercept

        try:
            model = sm.ZeroInflatedPoisson(endog, exog)
            instance.fitted_model = model.fit(disp=False, maxiter=1000)
            # Extract parameters
            instance.lambda_ = float(np.exp(instance.fitted_model.params[0]))
            # Zero-inflation probability is in the inflate params
            if hasattr(instance.fitted_model, "params_inflate"):
                logit_pi = instance.fitted_model.params_inflate[0]
                instance.pi = float(1.0 / (1.0 + np.exp(-logit_pi)))  # Inverse logit
            else:
                # Fallback estimation
                zeros_prop = float(np.mean(data == 0))
                expected_poisson_zeros = np.exp(-instance.lambda_)
                instance.pi = max(0.0, zeros_prop - expected_poisson_zeros)
        except Exception:
            # Fall back to manual estimation
            mean_data = float(np.mean(data))
            zeros_prop = float(np.mean(data == 0))

            # Initial lambda estimate
            instance.lambda_ = mean_data
            expected_poisson_zeros = np.exp(-instance.lambda_)
            instance.pi = max(0.001, min(0.999, zeros_prop - expected_poisson_zeros))

        return instance

    @classmethod
    def fit_parameters(cls, data: NDArray[np.integer[Any] | np.floating[Any]]) -> tuple[float, float]:
        """
        Fit Zero-Inflated Poisson distribution parameters to data.

        Parameters:
        -----------
        data : NDArray[np.integer[Any] | np.floating[Any]]
            Data to fit

        Returns:
        --------
        tuple[float, float] : Tuple containing (lambda, pi) parameters
        """
        fitted_dist = cls.fit(data)
        if fitted_dist.lambda_ is None or fitted_dist.pi is None:
            raise ValueError("Failed to fit Zero-Inflated Poisson distribution parameters")
        return (fitted_dist.lambda_, fitted_dist.pi)

    def pmf(self, k: NDArray[np.integer[Any] | np.floating[Any]]) -> NDArray[np.integer[Any] | np.floating[Any]]:
        """Probability mass function."""
        if self.lambda_ is None or self.pi is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")

        k_array = np.asarray(k, dtype=float)
        result = np.zeros_like(k_array, dtype=float)
        zero_mask: NDArray[np.bool_] = k_array == 0
        nonzero_mask: NDArray[np.bool_] = ~zero_mask

        if np.any(zero_mask):
            result[zero_mask] = self.pi + (1 - self.pi) * np.exp(-self.lambda_)

        if np.any(nonzero_mask):
            result[nonzero_mask] = (1 - self.pi) * stats.poisson.pmf(k_array[nonzero_mask], self.lambda_)

        return cast("NDArray[np.integer[Any] | np.floating[Any]]", result)

    def pdf(self, k: NDArray[np.integer[Any] | np.floating[Any]]) -> NDArray[np.integer[Any] | np.floating[Any]]:
        """PDF method for compatibility (same as PMF for discrete distribution)."""
        return self.pmf(k)

    def logpdf(self, k: NDArray[np.integer[Any] | np.floating[Any]]) -> NDArray[np.integer[Any] | np.floating[Any]]:
        """Log PDF method."""
        pmf_values = self.pmf(k)
        result = np.log(np.maximum(pmf_values, np.finfo(float).eps))  # Avoid log(0)
        return cast("NDArray[np.integer[Any] | np.floating[Any]]", result)

    def logpmf(self, k: NDArray[np.integer[Any] | np.floating[Any]]) -> NDArray[np.integer[Any] | np.floating[Any]]:
        """Log probability mass function."""
        return self.logpdf(k)

    def cdf(self, k: NDArray[np.integer[Any] | np.floating[Any]]) -> NDArray[np.integer[Any] | np.floating[Any]]:
        """Cumulative distribution function."""
        k_array = np.asarray(k, dtype=float)
        result = np.zeros_like(k_array, dtype=float)
        for i, ki in enumerate(k_array):
            result[i] = float(np.sum(self.pmf(np.arange(0, int(ki) + 1))))
        return cast("NDArray[np.integer[Any] | np.floating[Any]]", result)

    def rvs(self, size: int | tuple[int, ...] = 1, random_state: int = 42) -> NDArray[np.integer[Any]]:
        """Generate random variates."""
        if self.lambda_ is None or self.pi is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")

        rng = np.random.default_rng(random_state)
        is_zero = rng.random(size) < self.pi
        poisson_draws = rng.poisson(self.lambda_, size)
        result = np.where(is_zero, 0, poisson_draws)
        return cast("NDArray[np.integer[Any]]", result)

    def mean(self) -> float:
        """Mean of the distribution."""
        if self.lambda_ is None or self.pi is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return float((1 - self.pi) * self.lambda_)

    def var(self) -> float:
        """Variance of the distribution."""
        if self.lambda_ is None or self.pi is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        mean_val = (1 - self.pi) * self.lambda_
        return float(mean_val + self.pi * (1 - self.pi) * self.lambda_**2)


# Type for distribution classes and fitting functions
DistributionType = (
    type[stats.rv_continuous] |
    type[stats.rv_discrete] |
    type[StatsModelsPoissonDist] |
    type[StatsModelsNegativeBinomialDist] |
    type[StatsModelsZeroInflatedPoissonDist]
)
FitFunctionType = Callable[[NDArray[np.integer[Any] | np.floating[Any]]], tuple[float, ...]]


def get_distributions(metric_type: Literal["count", "time"]) -> dict[str, tuple[DistributionType, FitFunctionType]]:

    if metric_type == "count":
        return  {
            "negative_binomial": (
                StatsModelsNegativeBinomialDist,
                lambda x: StatsModelsNegativeBinomialDist.fit_parameters(x),
            ),
            "poisson": (
                StatsModelsPoissonDist,
                lambda x: StatsModelsPoissonDist.fit_parameters(x),
            ),
            "zero_inflated_poisson": (
                StatsModelsZeroInflatedPoissonDist,
                lambda x: StatsModelsZeroInflatedPoissonDist.fit_parameters(x),
            ),
        }

    return {
        "normal": (stats.norm, lambda x: stats.norm.fit(x)),
        "lognormal": (stats.lognorm, lambda x: stats.lognorm.fit(x, floc=0)),
        "gamma": (stats.gamma, lambda x: stats.gamma.fit(x, floc=0)),
        "weibull": (stats.weibull_min, lambda x: stats.weibull_min.fit(x, floc=0)),
    }
