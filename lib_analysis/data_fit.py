# mypy: disable-error-code="operator"

from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from lib_analysis.utils_distributions import get_distributions
from lib_analysis.utils_generic import is_falsy


class DistributionFitter:
    """Fit and evaluate multiple probability distributions."""

    MIN_SAMPLE_SIZE: int = 50
    DISTRIBUTION_CRITERIA: set[str] = {"aic", "bic", "cramer_von_mises"}

    def __init__(self, data_dict: dict[str, Any]) -> None:
        """
        Initialize the distribution fitter.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing data

        Returns:
        ---------
        None
        """
        self.data_dict = data_dict

    def fit_distributions(self) -> dict[str, Any]:
        """
        Fit multiple distributions to data and compute goodness-of-fit metrics.

        Returns:
        -------
        dict: Updated data dictionary
        """
        # Extract data from dictionary
        metric_config: dict[str, Any] = self.data_dict.get("metric_config", {})
        clean: dict[str, Any] = self.data_dict.get("clean", {})
        data: NDArray[np.integer[Any] | np.floating[Any]] = clean.get("data", np.array([]))
        metric_type: Literal["discrete", "continuous"] | None = metric_config.get("metric_type")
        distribution_best_criterion: Literal["aic", "bic", "cramer_von_mises"] | None =\
            self.data_dict.get("distribution_best_criterion", None)

        # Raise error if something is missing
        if any(map(is_falsy, (metric_config, clean, data, metric_type))):
            raise ValueError("---> The data dictionary does not contain all required parts.")

        # Validate metric type
        if metric_type not in {"discrete", "continuous"}:
            raise ValueError(f"---> Metric_type must be 'discrete' or 'continuous', got '{metric_type}'")

        # Validate distribution_best_criterion
        if distribution_best_criterion is not None and \
            distribution_best_criterion not in self.DISTRIBUTION_CRITERIA:
            raise ValueError(f"---> Distribution_best_criterion {distribution_best_criterion} is invalid")

        # Validate minimum sample size
        if data.size < self.MIN_SAMPLE_SIZE:
            raise ValueError(f"---> {self.MIN_SAMPLE_SIZE} measures are needed to fit distributions, got {data.size}")

        # Initialize results
        fitted_models: dict[str, Any] = {}
        failed_models: list[str] = []

        # Get distributions to fit
        distributions = get_distributions(metric_type)

        # Pre-sort data for efficiency
        sorted_data = np.sort(data)

        # Fit each distribution
        for dist_name, (dist_class, fit_func) in distributions.items():

            try:
                # Fit distribution
                parameters = fit_func(data)

                # Handle case where fitting fails and returns empty parameters
                if not parameters or len(parameters) == 0:
                    fitted_models[dist_name] = {
                        "parameters": None,
                        "goodness_of_fit": None,
                        "quantiles": None,
                    }
                    continue
                # Create distribution object
                dist_obj = dist_class(*parameters)

                # Compute metrics
                metrics = self._compute_metrics(dist_obj, data, sorted_data, data.size, metric_type)

                # Store results of fitted distribution
                fitted_models[dist_name] = {
                    "parameters": parameters,
                    "goodness_of_fit": metrics,
                    "quantiles": {
                        f"q{int(round(q*100,0))}": dist_obj.ppf(cast("float",q)) for q in np.arange(0.01, 1., 0.01)
                    },
                }

            # On fitting error
            except Exception as e:  # noqa: BLE001
                fitted_models[dist_name] = {
                    "parameters": None,
                    "goodness_of_fit": None,
                    "quantiles": None,
                }
                failed_models.append(f"{dist_name}: {e!s}")
                continue

        # Raise error if no distribution fitted successfully
        if len(failed_models) == len(distributions):
            raise ValueError(f"All distributions failed to fit. Errors: {failed_models}")

        # Determine best model
        best_model = self._get_best_model(fitted_models, distribution_best_criterion)

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
        """
        Determine the best fitting model based on specified criterion.

        Parameters
        ----------
        fitted_models : dict
            Dictionary of fitted distribution models

        results : dict
            Dictionary of goodness-of-fit metrics for each model

        criterion : str or None
            Selection criterion ('aic', 'bic', 'cramer_von_mises', or None for majority vote)

        Returns:
        -------
        dict: Best model information with 'name' and 'parameters' keys
        """
        # Filter out distributions with invalid values for all criteria
        valid_models = {
            name: data
            for name, data in fitted_models.items()
            if (all(data["goodness_of_fit"][crit] is not None and np.isfinite(data["goodness_of_fit"][crit])
                    for crit in self.DISTRIBUTION_CRITERIA))
        }

        # If there aare no valid models
        if not valid_models:
            return {"name": None, "parameters": None}

        # If specific criterion provided, use it
        if criterion is not None:

            # Raise error if criterion is invalid
            if criterion not in self.DISTRIBUTION_CRITERIA:
                raise ValueError(f"criterion must be one of {self.DISTRIBUTION_CRITERIA}, got '{criterion}'")

            # Compute best model via selected criterion
            best_model_name = min(valid_models.keys(), key=lambda x: valid_models[x][criterion])

            # Get parameters of best model
            best_parameters = fitted_models[best_model_name]["parameters"]

            return {"name": best_model_name, "parameters": best_parameters}

        # Use majority vote across all criteria
        return self._majority_vote_selection(fitted_models, valid_models)

    def _majority_vote_selection(
        self,
        fitted_models: dict[str, Any],
        valid_models: dict[str, Any],
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
        dict: Best model information with 'name' and 'parameters' keys
        """
        # If there is just one model
        if len(valid_models) == 1:
            name = next(iter(valid_models))
            return {"name": name, "parameters": fitted_models[name]["parameters"]}

        # Order by importance for tie-breaking
        criteria = ["bic", "aic", "cramer_von_mises"]

        # Get nodel names
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
                    True for crit in criteria
                    if valid_models[model_a]["goodness_of_fit"][crit] < valid_models[model_b]["goodness_of_fit"][crit]
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

        return {"name": best_model_name, "parameters": fitted_models[best_model_name]["parameters"]}

    def _break_tie(
        self,
        tied_models: list[str],
        valid_models: dict[str, Any],
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
        str: Name of the best model after tie-breaking
        """
        for crit in criteria:
            # Find best model for this criterion
            best_for_criterion = min(tied_models, key=lambda x: valid_models[x]["goodness_of_fit"][crit])

            # Check if this model is uniquely best for this criterion
            best_value = valid_models[best_for_criterion]["goodness_of_fit"][crit]
            other_values = [valid_models[model]["goodness_of_fit"][crit] for model in tied_models
                if model != best_for_criterion]

            # If uniquely best, return it
            if all(best_value < other_val for other_val in other_values):
                return best_for_criterion

        # If still tied, return first alphabetically (deterministic)
        return sorted(tied_models)[0]

    def _compute_metrics(
        self,
        dist_obj: Any,
        data: NDArray[np.integer[Any] | np.floating[Any]],
        sorted_data: NDArray[np.integer[Any] | np.floating[Any]],
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
            Type of metric ('discrete' or 'continuous')

        Returns:
        -------
        dict: Dictionary containing AIC, BIC, Cramér-von Mises and other fit statistics
        """
        # Log-likelihood
        try:
            if metric_type == "discrete":
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
        sorted_data: NDArray[np.integer[Any] | np.floating[Any]],
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
            Type of metric ('discrete' or 'continuous')

        Returns:
        -------
        tuple: Cramér-von Mises statistic and p-value (if available)
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
        sorted_data: NDArray[np.integer[Any] | np.floating[Any]],
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
            Type of metric ('discrete' or 'continuous')

        Returns:
        -------
        tuple: Cramér-von Mises statistic and p-value if successful, otherwise (None, None)
        """
        # Only try scipy for continuous distributions
        if metric_type != "continuous":
            return None, None

        try:
            # Use scipy's cramervonmises test
            result = stats.cramervonmises(sorted_data, dist_obj.cdf)
            return float(result.statistic), float(result.pvalue)
        except (AttributeError, ValueError, TypeError):
            pass

        return None, None
