from typing import Any


class ModelSelector:
    """Step 4: Select the best fitting distribution."""

    @staticmethod
    def select_best_model(
            data_dict: dict[str, Any],
            criterion: str = "aic",
        ) -> dict[str, Any]:
        """
        Select the best fitting distribution based on information criterion.

        Parameters:
        -----------
        data_dict : dict
            Data dictionary with fitted models
        criterion : str
            Selection criterion ('aic' or 'bic')

        Returns:
        --------
        dict : Updated data dictionary with best model selection
        """
        fit_results = data_dict["fit_results"]

        if criterion not in ["aic", "bic"]:
            raise ValueError("Criterion must be 'aic' or 'bic'")

        best_model_name = (
            min(fit_results.keys(),
                key=lambda x: fit_results[x][criterion])
        )

        best_model = data_dict["fitted_models"][best_model_name]

        data_dict.update({
            "best_model_name": best_model_name,
            "best_model": best_model,
            "selection_criterion": criterion,
        })

        return data_dict
