from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from lib.distributions import get_distributions


@dataclass
class FitResult:
    """Results from distribution fitting."""
    fitted_models: dict[str, dict[str, Any]]
    fit_results: dict[str, dict[str, float | int | None]]
    failed_fits: list[str]
    best_model: dict[str, Any]


class DistributionFitter:
    """Fit and evaluate multiple probability distributions."""

    MIN_SAMPLE_SIZE: int = 50

    def __init__(self, data_dict: dict[str, Any]) -> None:
        """
        Initialize the distribution fitter.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing data

        Returns:
        ----------
        None
        """
        self.data_dict = data_dict

    def _get_parameter_names(self, dist_name: str, dist_class: Any) -> list[str]:
        """
        Get statistically meaningful parameter names for a distribution.

        Parameters
        ----------
        dist_name : str
            Name of the distribution
        dist_class : Any
            Distribution class from scipy.stats

        Returns:
        -------
        list:
            List of parameter names with statistical meaning
        """
        # Statistically meaningful parameter name mappings
        param_mappings = {
            # Continuous distributions
            "norm": ["mean", "std_dev"],
            "expon": ["location", "rate_inv"],  # rate_inv = 1/rate
            "gamma": ["shape", "location", "scale"],
            "beta": ["alpha", "beta", "location", "scale"],
            "lognorm": ["sigma", "location", "scale"],  # sigma is shape parameter
            "weibull_min": ["shape", "location", "scale"],
            "uniform": ["lower_bound", "width"],
            "chi2": ["degrees_freedom", "location", "scale"],
            "f": ["dfn", "dfd", "location", "scale"],  # degrees of freedom numerator/denominator
            "t": ["degrees_freedom", "location", "scale"],
            "pareto": ["shape", "location", "scale"],
            "rayleigh": ["location", "scale"],
            "logistic": ["location", "scale"],
            "laplace": ["location", "scale"],
            "cauchy": ["location", "scale"],
            "gumbel_r": ["location", "scale"],
            "gumbel_l": ["location", "scale"],
            "exponweib": ["alpha", "beta", "location", "scale"],  # alpha=shape1, beta=shape2
            "genextreme": ["shape", "location", "scale"],
            "genpareto": ["shape", "location", "scale"],
            "invgamma": ["shape", "location", "scale"],
            "invweibull": ["shape", "location", "scale"],
            "kstwobign": [],  # No parameters for Kolmogorov-Smirnov
            "levy": ["location", "scale"],
            "levy_l": ["location", "scale"],
            "maxwell": ["location", "scale"],
            "nakagami": ["shape", "location", "scale"],
            "pearson3": ["skewness", "location", "scale"],
            "powerlaw": ["shape", "location", "scale"],
            "reciprocal": ["lower_bound", "upper_bound", "location", "scale"],
            "rice": ["shape", "location", "scale"],
            "truncexpon": ["shape", "location", "scale"],
            "truncnorm": ["lower_clip", "upper_clip", "location", "scale"],
            "vonmises": ["concentration", "location", "scale"],

            # Discrete distributions
            "poisson": ["rate"],  # lambda parameter
            "nbinom": ["failures", "success_prob"],  # n=number of failures, p=success probability
            "binom": ["trials", "success_prob"],  # n=number of trials, p=success probability
            "geom": ["success_prob"],  # p=success probability
            "hypergeom": ["population", "success_states", "draws"],  # M, K, N
            "randint": ["lower_bound", "upper_bound"],
            "zipf": ["exponent"],  # a parameter
            "planck": ["lambda"],  # lambda parameter
            "boltzmann": ["lambda", "N"],  # lambda and N parameters
            "dlaplace": ["shape"],  # a parameter
            "logser": ["shape"],  # p parameter
            "yulesimon": ["alpha"],  # alpha parameter
        }

        # Return known parameter names or try to infer from distribution
        if dist_name in param_mappings:
            return param_mappings[dist_name]

        # Try to get parameter names from the distribution's shapes
        if hasattr(dist_class, "shapes") and dist_class.shapes:
            shape_names = dist_class.shapes.split(", ")
            # Add location and scale for continuous distributions
            if hasattr(dist_class, "pdf"):  # Continuous
                return [*shape_names, "location", "scale"]
            # Discrete
            return shape_names

        # Default fallback based on distribution type
        if hasattr(dist_class, "pdf"):  # Continuous
            return ["location", "scale"]
        # Discrete
        return ["parameter"]

    def _format_parameters(self, dist_name: str, dist_class: Any, params: tuple) -> dict[str, dict[str, Any]]:
        """
        Format parameters as dictionary with meaningful names and values.

        Parameters
        ----------
        dist_name : str
            Name of the distribution

        dist_class : Any
            Distribution class from scipy.stats

        params : tuple
            Fitted parameter values

        Returns:
        -------
        dict
            Formatted parameters dictionary
        """
        if not params:
            return {}

        param_names = self._get_parameter_names(dist_name, dist_class)

        # Ensure we have the right number of parameter names
        if len(params) != len(param_names):
            # Adjust parameter names if needed
            if len(params) > len(param_names):
                param_names.extend([f"parameter_{i}" for i in range(len(param_names), len(params))])
            else:
                param_names = param_names[:len(params)]

        # Create formatted parameter dictionary
        formatted_params = {}
        for i, (name, value) in enumerate(zip(param_names, params, strict=False)):
            formatted_params[f"param_{i}"] = {
                "name": name,
                "value": float(value),
            }

        return formatted_params

    def _extract_parameter_values(self, formatted_params: dict[str, dict[str, Any]]) -> tuple:
        """
        Extract parameter values from formatted parameter dictionary.

        Parameters
        ----------
        formatted_params : dict
            Formatted parameters dictionary

        Returns:
        -------
        tuple
            Parameter values as tuple
        """
        if not formatted_params:
            return ()

        # Sort by parameter index to maintain order
        sorted_params = sorted(formatted_params.items(), key=lambda x: int(x[0].split("_")[1]))
        return tuple(param_info["value"] for _, param_info in sorted_params)

    def fit_distributions(self) -> FitResult:
        """
        Fit multiple distributions to data and compute goodness-of-fit metrics.

        Returns:
        -------
        FitResult
            Contains fitted models, goodness-of-fit metrics, and failed fits
        """
        # Extract data
        data = self.data_dict.get("analysis_data")
        metric_type = self.data_dict.get("metric_config").get("metric_type")
        distribution_best_criterion = self.data_dict.get("distribution_best_criterion", None)

        # Validate metric type
        if metric_type not in {"count", "time"}:
            raise ValueError(f"metric_type must be 'count' or 'time', got '{self.data_dict.get('metric_type')}'")

        # Validate minimum sample size
        if data.size < self.MIN_SAMPLE_SIZE:
            raise ValueError(f"{self.MIN_SAMPLE_SIZE} data points to fit theoretical distributions, got {data.size}")

        # Initialize results
        fitted_models: dict[str, dict[str, Any]] = {}
        fit_results: dict[str, dict[str, float | int | None]] = {}
        failed_fits: list[str] = []

        # Get distributions
        distributions = get_distributions(metric_type)

        # Pre-sort data for efficiency
        sorted_data = np.sort(data)
        n = len(data)

        # Fit each distribution
        for dist_name, (dist_class, fit_func) in distributions.items():

            try:
                # Fit distribution
                params = fit_func(data)

                # Handle case where fitting fails and returns empty params
                if not params or len(params) == 0:
                    fitted_models[dist_name] = {"parameters": {}}
                    fit_results[dist_name] = None
                    continue

                # Format parameters
                formatted_params = self._format_parameters(dist_name, dist_class, params)

                # Create distribution object
                dist_obj = dist_class(*params)

                # Compute metrics
                metrics = self._compute_metrics(dist_obj, data, sorted_data, n, metric_type)

                # Store results of fitted distribution
                fitted_models[dist_name] = {"parameters": formatted_params}
                fit_results[dist_name] = metrics

            # On fitting error
            except Exception as e:  # noqa: BLE001
                fitted_models[dist_name] = {"parameters": {}}
                fit_results[dist_name] = None
                failed_fits.append(f"{dist_name}: {e!s}")
                continue

        # Raise error if no distribution fitted successfully
        if len(failed_fits) == len(distributions):
            raise ValueError(f"All distributions failed to fit. Errors: {failed_fits}")

        # Determine best model
        best_model = self._get_best_model(fitted_models, fit_results, distribution_best_criterion)

        # Update data dict with fitted distributions
        self.data_dict["fitted_distribution"] = FitResult(
            fitted_models=fitted_models,
            fit_results=fit_results,
            failed_fits=failed_fits,
            best_model=best_model,
        )

        return self.data_dict

    def _get_best_model(
        self,
        fitted_models: dict[str, dict[str, Any]],
        fit_results: dict[str, dict[str, float | int | None]],
        criterion: str | None,
    ) -> dict[str, Any]:
        """
        Determine the best fitting model based on specified criterion.

        Parameters
        ----------
        fitted_models : dict
            Dictionary of fitted distribution models

        fit_results : dict
            Dictionary of goodness-of-fit metrics for each model

        criterion : str or None
            Selection criterion ('aic', 'bic', 'cramer_von_mises', or None for majority vote)

        Returns:
        -------
        dict
            Best model information with 'name' and 'params' keys
        """
        # Filter out distributions with invalid values for all criteria
        criteria = ["aic", "bic", "cramer_von_mises"]
        valid_models = {
            name: results
            for name, results in fit_results.items()
            if (results is not None and
                all(results[crit] is not None and np.isfinite(results[crit])
                    for crit in criteria))
        }

        # If there are no valid models
        if not valid_models:
            return {"name": None, "params": {}}

        # If specific criterion provided, use it
        if criterion is not None:

            # Raise error if criterion is invalid
            if criterion not in {"aic", "bic", "cramer_von_mises"}:
                raise ValueError(f"criterion must be one of {'aic', 'bic', 'cramer_von_mises', 'None'}")

            # Compute best model via selected criterion
            best_model_name = min(valid_models.keys(), key=lambda x: valid_models[x][criterion])

            # Get params of best model
            best_params = fitted_models[best_model_name]["parameters"]

            return {"name": best_model_name, "params": best_params}

        # Use majority vote across all criteria
        return self._majority_vote_selection(fitted_models, valid_models)

    def _majority_vote_selection(
        self,
        fitted_models: dict[str, dict[str, Any]],
        valid_models: dict[str, dict[str, float | int | None]],
    ) -> dict[str, Any]:
        """
        Select best model using majority vote across criteria.

        Parameters
        ----------
        fitted_models : dict
            Dictionary of fitted distribution models

        valid_models : dict
            Dictionary of valid models with their goodness-of-fit metrics

        Returns:
        -------
        dict
            Best model information with 'name' and 'params' keys
        """
        # If there is just one model
        if len(valid_models) == 1:
            name = next(iter(valid_models))
            return {"name": name, "params": fitted_models[name]["parameters"]}

        # Order by importance for tie-breaking
        criteria = ["bic", "aic", "cramer_von_mises"]

        # Get model names
        model_names = list(valid_models.keys())

        # Get number of valid models
        n_models = len(model_names)

        # Initialize win matrix for pairwise comparisons
        wins = dict.fromkeys(model_names, 0)

        # Perform pairwise comparisons
        for i in range(n_models):
            for j in range(i + 1, n_models):
                model_a, model_b = model_names[i], model_names[j]

                # Count model a wins for each criterion (lower is better)
                a_wins = sum(
                    1 for crit in criteria
                    if valid_models[model_a][crit] < valid_models[model_b][crit]
                )

                # Award win to model with majority of criteria
                # Ties don't award wins to either model
                if a_wins > len(criteria) / 2:
                    wins[model_a] += 1
                elif a_wins < len(criteria) / 2:
                    wins[model_b] += 1

        # Find model(s) with most pairwise wins
        max_wins = max(wins.values())
        winners = [name for name, win_count in wins.items() if win_count == max_wins]

        # Determine best model name
        best_model_name = winners[0] if len(winners) == 1 else self._break_tie(winners, valid_models, criteria)

        return {"name": best_model_name, "params": fitted_models[best_model_name]["parameters"]}

    def _break_tie(
        self,
        tied_models: list[str],
        valid_models: dict[str, dict[str, float | int | None]],
        criteria: list[str],
    ) -> str:
        """
        Break ties using hierarchical criterion preference (BIC > AIC > CvM).

        Parameters
        ----------
        tied_models : list[str]
            Names of models tied for best

        valid_models : dict
            Dictionary of valid models with their goodness-of-fit metrics

        criteria : list[str]
            List of criteria names ordered by preference

        Returns:
        -------
        str
            Name of the best model after tie-breaking
        """
        for criterion in criteria:
            # Find best model for this criterion
            best_for_criterion = min(tied_models, key=lambda x: valid_models[x][criterion])

            # Check if this model is uniquely best for this criterion
            best_value = valid_models[best_for_criterion][criterion]
            other_values = [valid_models[model][criterion] for model in tied_models
                if model != best_for_criterion]

            if all(best_value < other_val for other_val in other_values):
                return best_for_criterion

        # If still tied, return first alphabetically (deterministic)
        return sorted(tied_models)[0]

    def _compute_metrics(
        self,
        dist_obj: Any,
        data: npt.NDArray[np.floating],
        sorted_data: npt.NDArray[np.floating],
        n: int,
        metric_type: str,
    ) -> dict[str, float | int | None]:
        """
        Compute goodness-of-fit metrics for a fitted distribution.

        Parameters
        ----------
        dist_obj : Any
            Fitted distribution object

        data : ndarray
            Original data

        sorted_data : ndarray
            Pre-sorted data for efficiency

        n : int
            Number of data points

        metric_type : str
            Type of metric ('count' or 'time')

        Returns:
        -------
        dict
            Dictionary containing AIC, BIC, Cramér-von Mises and other fit statistics
        """
        # Log-likelihood
        try:
            if metric_type == "count":
                log_lik = float(np.sum(dist_obj.logpmf(data)))
            else:
                log_lik = float(np.sum(dist_obj.logpdf(data)))
        except (AttributeError, ValueError) as e:
            raise ValueError(f"Failed to compute log-likelihood: {e}") from e

        # Check for invalid log-likelihood
        if not np.isfinite(log_lik):
            raise ValueError("Log-likelihood is not finite")

        # Number of parameters
        if hasattr(dist_obj, "args"):
            k = len(dist_obj.args)
        elif hasattr(dist_obj, "kwds"):
            k = len(dist_obj.kwds)
        else:
            k = 2  # Default assumption (location, scale)

        # Information criteria
        aic = 2 * k - 2 * log_lik
        bic = k * np.log(n) - 2 * log_lik

        # Cramér-von Mises
        cvm_stat, cvm_pvalue = self._compute_cramer_von_mises(
            dist_obj, sorted_data, n, metric_type,
        )

        return {
            "log_likelihood": log_lik,
            "aic": aic,
            "bic": bic,
            "cramer_von_mises": cvm_stat,
            "cvm_pvalue": cvm_pvalue,
            "n_parameters": k,
        }

    def _compute_cramer_von_mises(
        self,
        dist_obj: Any,
        sorted_data: npt.NDArray[np.floating],
        n: int,
        metric_type: str,
    ) -> tuple[float, float | None]:
        """
        Compute Cramér-von Mises test statistic and p-value.

        Parameters
        ----------
        dist_obj : Any
            Fitted distribution object

        sorted_data : ndarray
            Pre-sorted data

        n : int
            Number of data points

        metric_type : str
            Type of metric ('count' or 'time')

        Returns:
        -------
        tuple[float, float | None]
            Cramér-von Mises statistic and p-value (if available)
        """
        try:
            # Try scipy's built-in test first (if available)
            cvm_stat, cvm_pvalue = self._try_scipy_cvm(sorted_data, dist_obj, metric_type)
            if cvm_stat is not None:
                return cvm_stat, cvm_pvalue

            # Manual calculation (works for both discrete and continuous)
            # Empirical CDF at sorted data points
            empirical_cdf = np.arange(1, n + 1) / n

            # Theoretical CDF at sorted data points
            theoretical_cdf = dist_obj.cdf(sorted_data)

            # Cramér-von Mises statistic
            cvm_stat = np.sum((empirical_cdf - theoretical_cdf) ** 2) + 1 / (12 * n)

            return float(cvm_stat), None

        except Exception:  # noqa: BLE001
            # Return NaN if calculation fails
            return float("nan"), None

    def _try_scipy_cvm(
        self,
        sorted_data: npt.NDArray[np.floating],
        dist_obj: Any,
        metric_type: str,
    ) -> tuple[float | None, float | None]:
        """
        Try to use scipy's Cramér-von Mises test for continuous distributions.

        Parameters
        ----------
        sorted_data : ndarray
            Pre-sorted data

        dist_obj : Any
            Fitted distribution object

        metric_type : str
            Type of metric ('count' or 'time')

        Returns:
        -------
        tuple[float | None, float | None]
            Cramér-von Mises statistic and p-value if successful, otherwise (None, None)
        """
        # Only try scipy for continuous distributions
        if metric_type != "time":
            return None, None

        try:
            # Use scipy's cramervonmises test
            result = stats.cramervonmises(sorted_data, dist_obj.cdf)
            return float(result.statistic), float(result.pvalue)
        except (AttributeError, ValueError, TypeError):
            pass

        return None, None
