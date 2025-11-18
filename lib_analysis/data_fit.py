# mypy: disable-error-code="operator"

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from lib_analysis.utils_distributions_continuous import get_continuous_distributions
from lib_analysis.utils_distributions_discrete import get_discrete_distributions
from lib_analysis.utils_generic import is_falsy

FitFunctionType = Callable[[NDArray[np.number[Any]]], tuple[float, ...]]


class DistributionFitter:
    """Fit and evaluate multiple probability distributions."""

    MIN_SAMPLE_SIZE: int = 50
    DISTRIBUTION_CRITERIA: set[str] = {"aic", "bic", "cramer_von_mises"}

    def __init__(self, data_dict: dict[str, Any]) -> None:
        """Initialize the distribution fitter.

        Args:
            data_dict: Dictionary containing data.
        """
        self.data_dict: dict[str, Any] = data_dict

    def fit_distributions(self) -> dict[str, Any]:
        """Fit multiple distributions to data and compute goodness-of-fit metrics.

        Returns:
            Updated data dictionary with fitted distribution results.

        Raises:
            ValueError: If the data dictionary is missing required components,
                if metric_type is invalid, if distribution_best_criterion is invalid,
                if insufficient sample size, or if all distributions fail to fit.
        """
        # Extract data from dictionary
        metric_config: dict[str, Any] = self.data_dict.get("metric_config", {})
        clean: dict[str, Any] = self.data_dict.get("clean", {})
        data: NDArray[np.number[Any]] = clean.get("data", np.array([]))
        metric_type: Literal["discrete", "continuous"] | None = metric_config.get("metric_type")
        distribution_best_criterion: Literal["aic", "bic", "cramer_von_mises"] | None =\
            self.data_dict.get("distribution_best_criterion", None)

        # Raise error if something is missing
        if any(map(is_falsy, (metric_config, clean, data, metric_type))):
            raise ValueError("---> The data dictionary does not contain all required parts.")

        # Validate metric type
        if metric_type not in {"discrete", "continuous"}:
            raise ValueError(f"---> Metric_type must be 'discrete' or 'continuous', got '{metric_type}'.")

        # Validate distribution_best_criterion
        if distribution_best_criterion is not None and \
            distribution_best_criterion not in self.DISTRIBUTION_CRITERIA:
            raise ValueError(f"---> Distribution_best_criterion {distribution_best_criterion} is invalid.")

        # Validate minimum sample size
        if data.size < self.MIN_SAMPLE_SIZE:
            raise ValueError(f"---> {self.MIN_SAMPLE_SIZE} measures are needed to fit distributions, got {data.size}.")

        # Initialize results
        fitted_models: dict[str, Any] = {}
        failed_models: list[str] = []

        # Get distributions
        distributions: dict[str, type[stats.rv_discrete | stats.rv_continuous]] =\
            get_continuous_distributions() if metric_type == "continuous" else get_discrete_distributions()

        # Pre-sort data for efficiency
        sorted_data: NDArray[np.number[Any]] = np.sort(data)

        # Iterate over distributions
        for dist_name, dist_class in distributions.items():

            try:
                # Get fitted parameters
                # For continuous distributions, fix location to 0 (metric data is non-negative)
                # For discrete distributions, use custom fit_parameters method
                parameters: tuple[float, ...] = dist_class.fit(data, floc=0)\
                    if metric_type == "continuous" else dist_class.fit_parameters(data)

                # Handle case where fitting fails and returns empty parameters
                if not parameters or len(parameters) == 0:
                    fitted_models[dist_name] = {
                        "parameters": None,
                        "goodness_of_fit": None,
                        "quantiles": None,
                    }
                    continue

                # Create distribution object
                dist_obj: stats.rv_discrete | stats.rv_continuous = dist_class(*parameters)

                # Compute metrics
                metrics: dict[str, float | None] =\
                    self._compute_metrics(dist_obj, data, sorted_data, data.size, metric_type)

                # Store results of fitted distribution
                fitted_models[dist_name] = {
                    "parameters": parameters,
                    "goodness_of_fit": metrics,
                }

            # On fitting error
            except Exception as e:  # noqa: BLE001
                fitted_models[dist_name] = {
                    "parameters": None,
                    "goodness_of_fit": None,
                }
                failed_models.append(f"{dist_name}: {e!s}")
                continue

        # Raise error if no distribution fitted successfully
        if len(failed_models) == len(distributions):
            raise ValueError(f"All distributions failed to fit. Errors: {failed_models}")

        # Determine best model
        best_model: dict[str, Any] = self._get_best_model(fitted_models, distribution_best_criterion)

        # Update data dict with fitted distributions
        self.data_dict["fit"] = {
            "fitted_models": fitted_models,
            "failed_models": failed_models,
            "best_model": best_model,
        }

        return self.data_dict

    def _get_best_model(
        self,
        fitted_models: dict[str, Any],
        criterion: str | None,
    ) -> dict[str, Any]:
        """Get the best fitting model based on specified criterion.

        Args:
            fitted_models: Dictionary of fitted distribution models.
            criterion: Selection criterion ('aic', 'bic', 'cramer_von_mises', or None
                for majority vote).

        Returns:
            Best model information with 'name' and 'parameters' keys.
        """
        # Filter out distributions with invalid values for all criteria
        valid_models: dict[str, Any] = {
            name: data
            for name, data in fitted_models.items()
                if (all(data["goodness_of_fit"][crit] is not None and np.isfinite(data["goodness_of_fit"][crit])
                    for crit in self.DISTRIBUTION_CRITERIA))
        }

        # If there are no valid models
        if not valid_models:
            return {
                "name": None,
                "parameters": None,
            }

        # If specific criterion provided, use it
        if criterion is not None:

            # Compute best model via selected criterion
            best_model_name: str = min(valid_models.keys(), key=lambda x: valid_models[x]["goodness_of_fit"][criterion])

            # Get parameters of best model
            best_parameters: tuple[float, ...] = fitted_models[best_model_name]["parameters"]

            return {
                "name": best_model_name,
                "parameters": best_parameters,
            }

        # Use majority vote across all criteria
        return self._majority_vote_selection(fitted_models, valid_models)

    def _majority_vote_selection(
        self,
        fitted_models: dict[str, Any],
        valid_models: dict[str, Any],
    ) -> dict[str, Any]:
        """Select the best model using majority vote across criteria.

        Args:
            fitted_models: Dictionary of fitted distribution models.
            valid_models: Dictionary of valid models with their goodness-of-fit metrics.

        Returns:
            Best model information with 'name' and 'parameters' keys.
        """
        # If there is just one model
        if len(valid_models) == 1:

            # Get next  valid model
            name: str = next(iter(valid_models))

            return {
                "name": name,
                "parameters": fitted_models[name]["parameters"],
            }

        # Order by importance for tie-breaking
        criteria: list[str] = ["bic", "aic", "cramer_von_mises"]

        # Get model names
        model_names: list[str] = list(valid_models.keys())

        # Initialize win matrix for pairwise comparisons
        wins: dict[str, int] = dict.fromkeys(model_names, 0)

        # Get number of valid models
        n_models: int = len(model_names)

        # Iterate over all pairs of models
        for i in range(n_models):

            # Iterate over models j > i to avoid duplicate comparisons
            for j in range(i + 1, n_models):

                # Get model names
                model_a: str
                model_b: str
                model_a, model_b = model_names[i], model_names[j]
                model_a_goodness_of_fit: dict[str, float | None] = valid_models[model_a]["goodness_of_fit"]
                model_b_goodness_of_fit: dict[str, float | None] = valid_models[model_b]["goodness_of_fit"]

                # Count model a wins for each criterion (lower is better)
                a_wins: int = sum(
                    True for crit in criteria
                        if model_a_goodness_of_fit[crit] < model_b_goodness_of_fit[crit]
                )

                # Award win to model with majority of criteria
                # Ties don't award wins to either model
                if a_wins > len(criteria) / 2:
                    wins[model_a] += 1
                elif a_wins < len(criteria) / 2:
                    wins[model_b] += 1

        # Find model(s) with most pairwise wins
        max_wins: int = max(wins.values())

        # Identify models with the highest number of wins
        # It is possible to have ties here
        winners: list[str] = [name for name, win_count in wins.items() if win_count == max_wins]

        # Determine best model name
        best_model_name: str = winners[0] if len(winners) == 1 else self._break_tie(winners, valid_models, criteria)

        return {
            "name": best_model_name,
            "parameters": fitted_models[best_model_name]["parameters"],
        }

    def _break_tie(
        self,
        tied_models: list[str],
        valid_models: dict[str, Any],
        criteria: list[str],
    ) -> str:
        """Break ties using hierarchical criterion preference (BIC > AIC > CvM).

        Args:
            tied_models: Names of models tied for best.
            valid_models: Dictionary of valid models with their goodness-of-fit metrics.
            criteria: List of criteria names ordered by preference.

        Returns:
            Name of the best model after tie-breaking.
        """
        # Iterate over criteria in order of preference
        for crit in criteria:

            # Find best model for this criterion
            best_for_criterion: str = min(tied_models, key=lambda x: valid_models[x]["goodness_of_fit"][crit])

            # Check if this model is uniquely best for this criterion
            best_value: float | None = valid_models[best_for_criterion]["goodness_of_fit"][crit]

            # Get other models' values for this criterion
            other_values: list[float | None] = [valid_models[model]["goodness_of_fit"][crit] for model in tied_models
                if model != best_for_criterion]

            # If uniquely best, return it
            if all(best_value < other_val for other_val in other_values):
                return best_for_criterion

        # If still tied, return first alphabetically
        return sorted(tied_models)[0]

    def _compute_metrics(
        self,
        dist_obj: Any,
        data: NDArray[np.number[Any]],
        sorted_data: NDArray[np.number[Any]],
        n: int,
        metric_type: str,
    ) -> dict[str, float | None]:
        """Compute goodness-of-fit metrics for a fitted distribution.

        Args:
            dist_obj: Fitted distribution object.
            data: Original data.
            sorted_data: Pre-sorted data for efficiency.
            n: Number of data points.
            metric_type: Type of metric ('discrete' or 'continuous').

        Returns:
            Dictionary containing AIC, BIC, Cramér-von Mises and other fit statistics.

        Raises:
            ValueError: If log-likelihood computation fails or is not finite.
        """
        # Log-likelihood
        try:
            if metric_type == "discrete":
                log_lik: float = float(np.sum(dist_obj.logpmf(data)))
            else:
                log_lik = float(np.sum(dist_obj.logpdf(data)))
        except (AttributeError, ValueError) as e:
            raise ValueError(f"Failed to compute log-likelihood: {e}") from e

        # Check for invalid log-likelihood
        if not np.isfinite(log_lik):
            raise ValueError("Log-likelihood is not finite")

        # Number of parameters
        k: int
        if hasattr(dist_obj, "args"):
            k = len(dist_obj.args)
        elif hasattr(dist_obj, "kwds"):
            k = len(dist_obj.kwds)
        else:
            # Default assumption if attributes not found (location and scale)
            k = 2

        # Information criteria
        aic: float = 2 * k - 2 * log_lik
        bic: float = k * np.log(n) - 2 * log_lik

        # Compute Cramér-von Mises
        cvm_stat: float
        cvm_pvalue: float | None
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
        sorted_data: NDArray[np.number[Any]],
        n: int,
        metric_type: str,
    ) -> tuple[float, float | None]:
        """Compute Cramér-von Mises test statistic and p-value.

        Args:
            dist_obj: Fitted distribution object.
            sorted_data: Pre-sorted data.
            n: Number of data points.
            metric_type: Type of metric ('discrete' or 'continuous').

        Returns:
            Cramér-von Mises statistic and p-value (if available).
        """
        try:
            # Try scipy's built-in test first (if available)
            cvm_stat: float | None
            cvm_pvalue: float | None
            cvm_stat, cvm_pvalue = self._try_scipy_cvm(sorted_data, dist_obj, metric_type)

            # If scipy's test is successful, use it
            if cvm_stat is not None:
                return cvm_stat, cvm_pvalue

            # Manual calculation (works for both discrete and continuous)
            # Empirical CDF at sorted data points
            empirical_cdf: NDArray[np.floating[Any]] = np.arange(1, n + 1) / n

            # Theoretical CDF at sorted data points
            theoretical_cdf: NDArray[np.floating[Any]] = dist_obj.cdf(sorted_data)

            # Cramér-von Mises statistic
            cvm_stat = np.sum((empirical_cdf - theoretical_cdf) ** 2) + 1 / (12 * n)

            return float(cvm_stat), None

        except Exception:  # noqa: BLE001
            # Return NaN if calculation fails
            return float("nan"), None

    def _try_scipy_cvm(
        self,
        sorted_data: NDArray[np.number[Any]],
        dist_obj: Any,
        metric_type: str,
    ) -> tuple[float | None, float | None]:
        """Try to use scipy's Cramér-von Mises test for continuous distributions.

        Args:
            sorted_data: Pre-sorted data.
            dist_obj: Fitted distribution object.
            metric_type: Type of metric ('discrete' or 'continuous').

        Returns:
            Cramér-von Mises statistic and p-value if successful, otherwise
            (None, None).
        """
        # Only try scipy for continuous distributions
        if metric_type != "continuous":
            return None, None

        try:
            # Use scipy's cramervonmises test
            result: stats._result_classes.CramerVonMisesResult = stats.cramervonmises(sorted_data, dist_obj.cdf)

            return float(result.statistic), float(result.pvalue)

        # If scipy's test fails, return None
        except (AttributeError, ValueError, TypeError):
            pass

        return None, None
