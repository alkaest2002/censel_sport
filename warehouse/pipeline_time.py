from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
import numpy as np
import orjson

from warehouse.distributions import BASE_DISTRIBUTIONS_CONTINUOUS
from warehouse.normative_table_validator import NormativeTableValidator

print(len(BASE_DISTRIBUTIONS_CONTINUOUS))
class PerformancePipeline:
    """
    Main pipeline class that orchestrates the entire analysis workflow.
    Receives data as numpy array directly.
    """
    def __init__(self, data_dict: dict[str, Any]) -> None:
        """Initialize the performance pipeline.

        Parameters:
        -----------
        data_dict : dict
            Initial data dictionary
        """
        self.results: dict[str, Any] = {}
        self.data_dict = data_dict

    def run_pipeline(self,
            clean_params: dict[str, Any] | None=None,
            fit_params: dict[str, Any] | None=None,
            validation_params: dict[str, Any] | None=None) -> None:
        """
        Run the complete performance analysis pipeline.

        Parameters:
        -----------
        data : np.ndarray
            Performance data as numpy array
        metric_config : dict
            Metric configuration (name, units, higher_is_better)
        clean_params : dict, optional
            Parameters for data cleaning
        fit_params : dict, optional
            Parameters for distribution fitting
        validation_params : dict, optional
            Parameters for validation

        Returns:
            None
        """

        # Set default parameters
        clean_params = clean_params or {}
        fit_params = fit_params or {}
        validation_params = validation_params or {}

        print("="*70)
        print(f"PERFORMANCE ANALYSIS PIPELINE: {metric_config['name']}")
        print("="*70)

        # Step 1: Load Data
        print("\nStep 1: Preparing data structure...")
        self.data = self._load_data(data, metric_config)
        # # Step 2: Clean Data
        # print("\nStep 2: Cleaning data...")
        # self.data = DataCleaner.clean_data(self.data, **clean_params)
        # # Step 3: Fit Distributions
        # print("\nStep 3: Fitting distributions...")
        # self.data = DistributionFitter.fit_distributions(self.data, BASE_DISTRIBUTIONS_CONTINUOUS, **fit_params)
        # # Step 4: Select Best Model
        # print("\nStep 4: Selecting best model...")
        # self.data = ModelSelector.select_best_model(self.data)
        # # Step 5: Calculate Percentiles
        # print("\nStep 5: Calculating percentiles...")
        # self.data = PercentileCalculator.calculate_percentiles(self.data)
        # # Step 6: Create Normative Table
        # print("\nStep 6: Creating normative table...")
        # self.data = NormativeTableCreator.create_normative_table(self.data)
        # # Step 7: Validate Table
        # print("\nStep 7: Validating normative table...")
        # self._validate_normative_table(validation_params)
        # print("Validation completed")

        # Generate summary
        self._generate_summary()

        with Path("./results").open("w") as fout:
            fout.write(orjson.dumps(self.data).decode("utf-8"))

    def _validate_normative_table(self, validation_params: dict[str, Any]) -> None:
        """
        Perform normative table validation.

        Parameters:
        -----------
        validation_params : dict
            Validation parameters
        """
        # Bootstrap validation
        n_bootstrap = validation_params.get("n_bootstrap", 500)
        self.data = NormativeTableValidator.bootstrap_validation(self.data, n_bootstrap)

        # Monte Carlo validation
        n_montecarlo = validation_params.get("n_montecarlo", 500)
        self.data = NormativeTableValidator.montecarlo_validation(self.data, n_montecarlo)

    def _generate_summary(self) -> None:
        """Generate and display pipeline summary."""
        print("\n" + "="*70)
        print("PIPELINE SUMMARY")
        print("="*70)

        metric_config = self.data["metric_config"]
        print(f"Metric: {metric_config['name']} ({metric_config['units']})")
        print(f"Direction: {'Higher' if metric_config['higher_is_better'] else 'Lower'} is better")

        clean_data = self.data["clean_data"]
        print(f"Sample size: {len(clean_data)}")
        print(f"Mean: {np.mean(clean_data):.2f}")
        print(f"Std: {np.std(clean_data):.2f}")
        print(f"Range: {np.min(clean_data):.2f} - {np.max(clean_data):.2f}")

        print(f"Best distribution: {self.data['best_model_name']}")

        # Display normative table
        print("\nNormative Table:")
        print("-" * 80)
        table = self.data["normative_table"]
        for _, row in table.iterrows():
            level = int(row["Performance_Level"])
            desc = row["Description"]
            pct = row["Sample_Percentage"]
            print(f"Level {level} ({desc:<15}): {pct:5.1f}% of sample")

        # Validation summary
        if "bootstrap_results" in self.data:
            bootstrap_stability = self._calculate_validation_stability(self.data["bootstrap_results"])
            print(f"\nBootstrap stability (avg CV): {bootstrap_stability:.1f}%")

        if "montecarlo_results" in self.data:
            montecarlo_stability = self._calculate_validation_stability(self.data["montecarlo_results"])
            print(f"Monte Carlo stability (avg CV): {montecarlo_stability:.1f}%")

    def _calculate_validation_stability(self, validation_results: np.ndarray) -> float:
        """Calculate average coefficient of variation across performance levels."""
        cvs = []
        for level in range(7):
            level_data = validation_results[:, level]
            mean_val = np.mean(level_data)
            if mean_val > 0:
                cv = (np.std(level_data) / mean_val) * 100
                cvs.append(cv)
        return np.mean(cvs) if cvs else 0.0

    def get_performance_level(self, value: float) -> tuple[int | None, str]:
        """
        Classify a performance value into a normative level.

        Parameters:
        -----------
        value : float
            Performance value to classify

        Returns:
        --------
        tuple : (level, description)
        """
        if self.data is None or "normative_table" not in self.data:
            raise ValueError("Pipeline must be run first")

        normative_table = self.data["normative_table"]
        higher_is_better = self.data["metric_config"]["higher_is_better"]

        level = NormativeTableValidator.classify_single_value(value, normative_table, higher_is_better)
        if level is not None:
            description = normative_table.iloc[level]["Description"]
            return level, description

        return None, "Not classified"

    def get_data_source_info(self) -> dict[str, Any]:
        """
        Get information about the data source used in the pipeline.

        Returns:
        --------
        dict : Data source information
        """
        if self.data is None:
            raise ValueError("Pipeline must be run first")

        metadata = self.data.get("metadata", {})
        return {
            "source_type": metadata.get("source_type", "unknown"),
            "total_records": metadata.get("total_records", 0),
            "valid_records": metadata.get("valid_records", 0),
            "invalid_records": metadata.get("invalid_records", 0),
        }

    def plot_results(self) -> None:
        """Generate visualization of pipeline results."""
        if self.data is None:
            raise ValueError("Pipeline must be run first")

        _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Data distribution with fitted model
        clean_data = self.data["clean_data"]
        ax1.hist(clean_data, bins=50, density=True, alpha=0.7, color="lightblue",
                edgecolor="black", label="Data")

        x = np.linspace(clean_data.min(), clean_data.max(), 1000)
        best_model = self.data["best_model"]["distribution"]
        higher_is_better = self.data["metric_config"]["higher_is_better"]

        if higher_is_better:
            max_val = np.max(clean_data)
            x_analysis = max_val + 1 - x
            y = best_model.pdf(x_analysis)
        else:
            y = best_model.pdf(x)

        ax1.plot(x, y, "r-", linewidth=2, label=f'Best fit: {self.data["best_model_name"]}')
        ax1.set_xlabel(f'{self.data["metric_config"]["name"]} ({self.data["metric_config"]["units"]})')
        ax1.set_ylabel("Density")
        ax1.set_title("Data Distribution with Best Fit")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Normative levels distribution
        table = self.data["normative_table"]
        colors = ["red", "orange", "yellow", "lightgreen", "green", "blue", "purple"]
        ax2.bar(table["Performance_Level"], table["Sample_Percentage"],
            color=colors, alpha=0.7, edgecolor="black")
        ax2.set_xlabel("Performance Level")
        ax2.set_ylabel("Percentage of Sample")
        ax2.set_title("Normative Level Distribution")
        ax2.set_xticks(range(7))
        ax2.grid(True, alpha=0.3, axis="y")

        # 3. Model comparison (AIC values)
        if len(self.data["fit_results"]) > 1:
            models = list(self.data["fit_results"].keys())
            aics = [self.data["fit_results"][model]["aic"] for model in models]
            ax3.bar(models, aics, alpha=0.7, color="skyblue", edgecolor="black")
            ax3.set_ylabel("AIC Value")
            ax3.set_title("Model Comparison (Lower AIC is Better)")
            ax3.grid(True, alpha=0.3, axis="y")
            plt.setp(ax3.get_xticklabels(), rotation=45)

        # 4. Validation results comparison
        if "bootstrap_results" in self.data and "montecarlo_results" in self.data:
            bootstrap_means = np.mean(self.data["bootstrap_results"], axis=0)
            montecarlo_means = np.mean(self.data["montecarlo_results"], axis=0)

            x_pos = np.arange(7)
            width = 0.35

            ax4.bar(x_pos - width/2, bootstrap_means, width, label="Bootstrap",
                alpha=0.7, color="lightcoral")
            ax4.bar(x_pos + width/2, montecarlo_means, width, label="Monte Carlo",
                alpha=0.7, color="lightsteelblue")

            ax4.set_xlabel("Performance Level")
            ax4.set_ylabel("Mean Percentage")
            ax4.set_title("Validation Results Comparison")
            ax4.set_xticks(x_pos)
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.show()
