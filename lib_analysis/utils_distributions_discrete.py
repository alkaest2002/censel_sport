from collections.abc import Callable
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy import stats
import statsmodels.api as sm

FitFunctionType = Callable[[NDArray[np.number[Any]]], tuple[float, ...]]

# ruff: noqa: BLE001
# ruff: noqa: SLF001


class StatsModelsGeometricDist(stats.rv_discrete):
    """Geometric distribution using scipy stats with statsmodels-style interface."""

    def __init__(self, p: float | None = None) -> None:
        self.p: float | None = p

    @classmethod
    def fit(cls, data: NDArray[np.number[Any]]) -> "StatsModelsGeometricDist":
        """
        Fit Geometric distribution to data.

        The geometric distribution models the number of trials needed to get the first success.

        Parameters:
        -----------
        data : NDArray[np.number[Any]]
            Data to fit (should be positive integers)

        Returns:
        --------
        StatsModelsGeometricDist : Fitted distribution instance
        """
        instance = cls()

        # Method of moments estimator for geometric distribution
        # For geometric distribution: E[X] = 1/p, so p = 1/mean
        data_array = np.asarray(data, dtype=float)

        # Filter out non-positive values as geometric distribution starts from 1
        valid_data = data_array[data_array > 0]

        if len(valid_data) == 0:
            raise ValueError("Geometric distribution requires positive integer data")

        mean_val = float(np.mean(valid_data))

        if mean_val <= 0:
            raise ValueError("Mean must be positive for geometric distribution")

        instance.p = min(0.999, max(0.001, 1.0 / mean_val))  # Clamp to avoid edge cases

        return instance

    @classmethod
    def fit_parameters(cls, data: NDArray[np.number[Any]]) -> tuple[float]:
        """
        Fit Geometric distribution parameters to data.

        Parameters:
        -----------
        data : NDArray[np.number[Any]]
            Data to fit

        Returns:
        --------
        tuple[float] : Tuple containing (p,) parameter
        """
        fitted_dist = cls.fit(data)
        if fitted_dist.p is None:
            raise ValueError("Failed to fit Geometric distribution parameters")
        return (fitted_dist.p,)

    def pmf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Probability mass function."""
        if self.p is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return cast("NDArray[np.number[Any]]", stats.geom.pmf(k, self.p))

    def logpmf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Log probability mass function."""
        if self.p is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return cast("NDArray[np.number[Any]]", stats.geom.logpmf(k, self.p))

    def pdf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """PDF method for compatibility (same as PMF for discrete distribution)."""
        return self.pmf(k)

    def logpdf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Log PDF method for compatibility (same as log PMF for discrete distribution)."""
        return self.logpmf(k)

    def cdf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Cumulative distribution function."""
        if self.p is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return cast("NDArray[np.number[Any]]", stats.geom.cdf(k, self.p))

    def ppf(self, q: NDArray[np.number[Any]] | float) -> NDArray[np.integer[Any]] | int:
        """Percent point function (quantile function)."""
        if self.p is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return cast("NDArray[np.integer[Any]] | int", stats.geom.ppf(q, self.p))

    def rvs(self, size: int | tuple[int, ...] | None = 1, random_state: int | None = 42) -> NDArray[np.integer[Any]]:
        """Generate random variates."""
        if self.p is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return cast("NDArray[np.integer[Any]]", stats.geom.rvs(self.p, size=size, random_state=random_state))

    def mean(self) -> float:
        """Mean of the distribution."""
        if self.p is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return float(1.0 / self.p)

    def var(self) -> float:
        """Variance of the distribution."""
        if self.p is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return float((1.0 - self.p) / (self.p**2))


class StatsModelsBinomialDist(stats.rv_discrete):
    """Binomial distribution using scipy stats with statsmodels-style interface."""

    def __init__(self, n: int | None = None, p: float | None = None) -> None:
        self.n: int | None = n
        self.p: float | None = p

    @classmethod
    def fit(cls, data: NDArray[np.number[Any]], n: int | None = None) -> "StatsModelsBinomialDist":
        """
        Fit Binomial distribution to data.

        Parameters:
        -----------
        data : NDArray[np.number[Any]]
            Data to fit (should be non-negative integers)
        n : int | None
            Number of trials. If None, will be estimated as max(data)

        Returns:
        --------
        StatsModelsBinomialDist : Fitted distribution instance
        """
        instance = cls()

        data_array = np.asarray(data, dtype=int)

        # Filter out negative values
        valid_data = data_array[data_array >= 0]

        if len(valid_data) == 0:
            raise ValueError("Binomial distribution requires non-negative integer data")

        # Estimate n if not provided
        if n is None:
            instance.n = int(np.max(valid_data))
        else:
            instance.n = n

        if instance.n <= 0:
            raise ValueError("Number of trials (n) must be positive")

        # Estimate p using method of moments: p = mean / n
        mean_val = float(np.mean(valid_data))
        instance.p = min(0.999, max(0.001, mean_val / instance.n))

        return instance

    @classmethod
    def fit_parameters(
        cls, data: NDArray[np.number[Any]], n: int | None = None) -> tuple[int, float]:
        """
        Fit Binomial distribution parameters to data.

        Parameters:
        -----------
        data : NDArray[np.number[Any]]
            Data to fit
        n : int | None
            Number of trials. If None, will be estimated as max(data)

        Returns:
        --------
        tuple[int, float] : Tuple containing (n, p) parameters
        """
        fitted_dist = cls.fit(data, n=n)
        if fitted_dist.n is None or fitted_dist.p is None:
            raise ValueError("Failed to fit Binomial distribution parameters")
        return (fitted_dist.n, fitted_dist.p)

    def pmf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Probability mass function."""
        if self.n is None or self.p is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return cast("NDArray[np.number[Any]]", stats.binom.pmf(k, self.n, self.p))

    def logpmf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Log probability mass function."""
        if self.n is None or self.p is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return cast("NDArray[np.number[Any]]", stats.binom.logpmf(k, self.n, self.p))

    def pdf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """PDF method for compatibility (same as PMF for discrete distribution)."""
        return self.pmf(k)

    def logpdf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Log PDF method for compatibility (same as log PMF for discrete distribution)."""
        return self.logpmf(k)

    def cdf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Cumulative distribution function."""
        if self.n is None or self.p is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return cast("NDArray[np.number[Any]]", stats.binom.cdf(k, self.n, self.p))

    def ppf(self, q: NDArray[np.number[Any]] | float) -> NDArray[np.integer[Any]] | int:
        """Percent point function (quantile function)."""
        if self.n is None or self.p is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return cast("NDArray[np.integer[Any]] | int", stats.binom.ppf(q, self.n, self.p))

    def rvs(self, size: int | tuple[int, ...] | None = 1, random_state: int | None = 42) -> NDArray[np.integer[Any]]:
        """Generate random variates."""
        if self.n is None or self.p is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return cast("NDArray[np.integer[Any]]", stats.binom.rvs(self.n, self.p, size=size, random_state=random_state))

    def mean(self) -> float:
        """Mean of the distribution."""
        if self.n is None or self.p is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return float(self.n * self.p)

    def var(self) -> float:
        """Variance of the distribution."""
        if self.n is None or self.p is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return float(self.n * self.p * (1 - self.p))


class StatsModelsPoissonDist(stats.rv_discrete):
    """Poisson distribution using statsmodels."""

    def __init__(self, lambda_: float | None = None) -> None:
        self._lambda: float | None = lambda_

    @classmethod
    def fit(cls, data: NDArray[np.number[Any]]) -> "StatsModelsPoissonDist":
        """
        Fit Poisson distribution to data using statsmodels.

        Parameters:
        -----------
        data : NDArray[np.number[Any]]
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
            fitted_model = model.fit(disp=False, maxiter=1000)
            # Extract lambda from the fitted intercept
            instance._lambda = float(np.exp(fitted_model.params[0]))
        except Exception:
            # Fall back to simple MLE (mean)
            instance._lambda = float(np.mean(data))

        return instance

    @classmethod
    def fit_parameters(cls, data: NDArray[np.number[Any]]) -> tuple[float]:
        """
        Fit Poisson distribution parameters to data.

        Parameters:
        -----------
        data : NDArray[np.number[Any]]
            Data to fit

        Returns:
        --------
        tuple[float] : Tuple containing (lambda,) parameter
        """
        fitted_dist = cls.fit(data)
        if fitted_dist._lambda is None:
            raise ValueError("Failed to fit Poisson distribution parameters")
        return (fitted_dist._lambda,)

    def pmf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Probability mass function."""
        if self._lambda is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return cast("NDArray[np.number[Any]]", stats.poisson.pmf(k, self._lambda))

    def logpmf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Log probability mass function."""
        if self._lambda is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return cast("NDArray[np.number[Any]]", stats.poisson.logpmf(k, self._lambda))

    def pdf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """PDF method for compatibility (same as PMF for discrete distribution)."""
        return self.pmf(k)

    def logpdf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Log PDF method for compatibility (same as log PMF for discrete distribution)."""
        return self.logpmf(k)

    def cdf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Cumulative distribution function."""
        if self._lambda is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return cast("NDArray[np.number[Any]]", stats.poisson.cdf(k, self._lambda))

    def ppf(self, q: NDArray[np.number[Any]] | float) -> NDArray[np.integer[Any]] | int:
        """Percent point function (quantile function)."""
        if self._lambda is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")
        return cast("NDArray[np.integer[Any]] | int", stats.poisson.ppf(q, self._lambda))

    def rvs(self, size: int | tuple[int, ...] | None = 1, random_state: int | None = 42) -> NDArray[np.integer[Any]]:
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


class StatsModelsNegativeBinomialDist(stats.rv_discrete):
    """Negative Binomial distribution using statsmodels."""

    def __init__(self, mu: float | None = None, alpha: float | None = None) -> None:
        self.mu: float | None = mu
        self.alpha: float | None = alpha

    @classmethod
    def fit(cls, data: NDArray[np.number[Any]]) -> "StatsModelsNegativeBinomialDist":
        """
        Fit Negative Binomial distribution to data using statsmodels.

        Parameters:
        -----------
        data : NDArray[np.number[Any]]
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
            fitted_model = model.fit(disp=False, maxiter=1000)
            # Extract parameters
            instance.mu = float(np.exp(fitted_model.params[0]))
            instance.alpha = float(fitted_model.params[1])  # Dispersion parameter
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
    def fit_parameters(cls, data: NDArray[np.number[Any]]) -> tuple[float, float]:
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

    def pmf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Probability mass function."""
        n, p = self._convert_to_scipy_params()
        return cast("NDArray[np.number[Any]]", stats.nbinom.pmf(k, n, p))

    def logpmf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Log probability mass function."""
        n, p = self._convert_to_scipy_params()
        return cast("NDArray[np.number[Any]]", stats.nbinom.logpmf(k, n, p))

    def pdf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """PDF method for compatibility (same as PMF for discrete distribution)."""
        return self.pmf(k)

    def logpdf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Log PDF method for compatibility (same as log PMF for discrete distribution)."""
        return self.logpmf(k)

    def cdf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Cumulative distribution function."""
        n, p = self._convert_to_scipy_params()
        return cast("NDArray[np.number[Any]]", stats.nbinom.cdf(k, n, p))

    def ppf(self, q: NDArray[np.number[Any]] | float) -> NDArray[np.integer[Any]] | int:
        """Percent point function (quantile function)."""
        n, p = self._convert_to_scipy_params()
        return cast("NDArray[np.integer[Any]] | int", stats.nbinom.ppf(q, n, p))

    def rvs(self, size: int | tuple[int, ...] | None = 1, random_state: int | None = 42) -> NDArray[np.integer[Any]]:
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


class StatsModelsZeroInflatedPoissonDist(stats.rv_discrete):
    """Zero-Inflated Poisson distribution using statsmodels."""

    def __init__(self, lambda_: float | None = None, pi: float | None = None) -> None:
        self.lambda_: float | None = lambda_
        self.pi: float | None = pi

    @classmethod
    def fit(cls, data: NDArray[np.number[Any]]) -> "StatsModelsZeroInflatedPoissonDist":
        """
        Fit Zero-Inflated Poisson distribution to data using statsmodels.

        Parameters:
        -----------
        data : NDArray[np.number[Any]]
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
            fitted_model = model.fit(disp=False, maxiter=1000)
            # Extract parameters
            instance.lambda_ = float(np.exp(fitted_model.params[0]))
            # Zero-inflation probability is in the inflate params
            if hasattr(fitted_model, "params_inflate"):
                logit_pi = fitted_model.params_inflate[0]
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
    def fit_parameters(cls, data: NDArray[np.number[Any]]) -> tuple[float, float]:
        """
        Fit Zero-Inflated Poisson distribution parameters to data.

        Parameters:
        -----------
        data : NDArray[np.number[Any]]
            Data to fit

        Returns:
        --------
        tuple[float, float] : Tuple containing (lambda, pi) parameters
        """
        fitted_dist = cls.fit(data)
        if fitted_dist.lambda_ is None or fitted_dist.pi is None:
            raise ValueError("Failed to fit Zero-Inflated Poisson distribution parameters")
        return (fitted_dist.lambda_, fitted_dist.pi)

    def pmf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
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

        return cast("NDArray[np.number[Any]]", result)

    def pdf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """PDF method for compatibility (same as PMF for discrete distribution)."""
        return self.pmf(k)

    def logpdf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Log PDF method."""
        pmf_values = self.pmf(k)
        result = np.log(np.maximum(pmf_values, np.finfo(float).eps))  # Avoid log(0)
        return cast("NDArray[np.number[Any]]", result)

    def logpmf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Log probability mass function."""
        return self.logpdf(k)

    def cdf(self, k: NDArray[np.number[Any]]) -> NDArray[np.number[Any]]:
        """Cumulative distribution function."""
        k_array = np.asarray(k, dtype=float)
        result = np.zeros_like(k_array, dtype=float)
        for i, ki in enumerate(k_array):
            result[i] = float(np.sum(self.pmf(np.arange(0, int(ki) + 1))))
        return cast("NDArray[np.number[Any]]", result)

    def ppf(self, q: NDArray[np.number[Any]] | float) -> NDArray[np.integer[Any]] | int:
        """
        Percent point function (quantile function) for Zero-Inflated Poisson.

        For discrete distributions, this finds the smallest integer k such that CDF(k) >= q.
        """
        if self.lambda_ is None or self.pi is None:
            raise ValueError("Distribution not fitted. Use .fit() first.")

        q_array = np.asarray(q, dtype=float)
        is_scalar = q_array.ndim == 0

        if is_scalar:
            q_array = np.array([q_array])

        result = np.zeros(len(q_array), dtype=int)

        for i, qi in enumerate(q_array):
            if qi < 0 or qi > 1:
                result[i] = np.nan
                continue

            if qi == 0:
                result[i] = 0
                continue

            # Start from 0 and find the first k where CDF(k) >= q
            k = 0
            max_k = max(100, int(self.lambda_ + 5 * np.sqrt(self.lambda_)))  # Reasonable upper bound

            while k <= max_k:
                cdf_val = float(self.cdf(np.array([k]))[0])
                if cdf_val >= qi:
                    result[i] = k
                    break
                k += 1
            else:
                # If we didn't find a solution within max_k, use max_k
                result[i] = max_k

        if is_scalar:
            return int(result[0])
        return cast("NDArray[np.integer[Any]]", result)

    def rvs(self, size: int | tuple[int, ...] | None = 1, random_state: int | None = 42) -> NDArray[np.integer[Any]]:
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


def get_discrete_distributions(
    ) -> dict[str, stats.rv_continuous | stats.rv_discrete]:
    """
    Get a mapping of distribution names to their corresponding SciPy distribution objects.

    Returns:
    --------
    dict: Mapping of distribution names to distribution classes
    """
    return {
        "geometric": StatsModelsGeometricDist,
        "binomial": StatsModelsBinomialDist,
        "negative_binomial": StatsModelsNegativeBinomialDist,
        "poisson": StatsModelsPoissonDist,
        "zero_inflated_poisson": StatsModelsZeroInflatedPoissonDist,
    }
