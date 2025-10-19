from typing import Literal


def fit_distributions(
    data: ArrayLike,
    data_type: Literal["continuous", "count"] = "continuous",
) -> dict[str, str | dict[str, Any] | dict[str, dict[str, Any]]]:
    """
    Fit multiple distributions and select the best one.

    Parameters:
    -----------
    data : array-like
        Observed data
    data_type : str
        'continuous' for running times, 'count' for sit-ups

    Returns:
    --------
    dict : Best fit distribution info and all fitted distributions
    """
    data = np.array(data)
    results: dict[str, dict[str, Any]] = {}

    if data_type == "continuous":
        distributions: dict[str, type] = {
            "lognormal": stats.lognorm,
            "gamma": stats.gamma,
            "weibull": stats.weibull_min,
            "normal": stats.norm,
        }
    else:  # count data
        distributions = {
            "poisson": stats.poisson,
            "nbinom": stats.nbinom,
        }

    print(f"\nFitting {data_type} distributions...")

    for name, dist in distributions.items():
        try:
            # Fit distribution
            if name == "poisson":
                params = (np.mean(data),)
                fitted = dist(*params)
            elif name == "nbinom":
                # Negative binomial requires special fitting
                mean = np.mean(data)
                var = np.var(data)
                if var > mean:
                    # Method of moments for n and p
                    n = mean**2 / (var - mean)
                    p = mean / var
                    params = (n, p)
                    fitted = dist(n, p)
                else:
                    continue  # Skip if not overdispersed
            else:
                params = dist.fit(data)
                fitted = dist(*params)

            # Compute goodness of fit metrics
            # Log-likelihood
            log_likelihood = np.sum(fitted.logpdf(data) if data_type == "continuous"
                                   else fitted.logpmf(data))

            # AIC and BIC
            k = len(params)
            n = len(data)
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood

            # Kolmogorov-Smirnov test
            if data_type == "continuous":
                ks_stat, ks_pval = stats.kstest(data, lambda x: fitted.cdf(x))
            else:
                # For discrete data, use chi-square goodness of fit
                unique_vals = np.unique(data)
                observed_freq = np.array([np.sum(data == val) for val in unique_vals])
                expected_freq = np.array([len(data) * fitted.pmf(val) for val in unique_vals])
                # Avoid division by zero
                expected_freq = np.maximum(expected_freq, 1e-10)
                chi2_stat = np.sum((observed_freq - expected_freq)**2 / expected_freq)
                ks_stat, ks_pval = chi2_stat, stats.chi2.sf(chi2_stat, len(unique_vals) - k - 1)

            results[name] = {
                "distribution": fitted,
                "params": params,
                "aic": aic,
                "bic": bic,
                "log_likelihood": log_likelihood,
                "ks_stat": ks_stat,
                "ks_pval": ks_pval,
            }

            print(f"  {name:12s}: AIC={aic:.2f}, BIC={bic:.2f}, KS p-value={ks_pval:.4f}")

        except Exception as e:
            print(f"  {name:12s}: Failed to fit ({e!s})")
            continue

    # Select best distribution (lowest BIC)
    if results:
        best_name = min(results.keys(), key=lambda x: results[x]["bic"])
        print(f"\n  Best fit: {best_name} (lowest BIC)")

        return {
            "best_distribution": best_name,
            "best_fit": results[best_name],
            "all_fits": results,
        }
    raise ValueError("No distributions could be fitted successfully")


def monte_carlo_validation(
    bootstrap_table: pd.DataFrame,
    best_fit_info: dict[str, Any],
    data: ArrayLike,
    percentiles: list[int | float],
    n_simulations: int = 1000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[int | float, list[float]]]:
    """
    Validate bootstrap percentiles using Monte Carlo simulation from fitted distribution.

    Parameters:
    -----------
    bootstrap_table : pd.DataFrame
        Bootstrap-derived normative table
    best_fit_info : dict
        Output from fit_distributions
    data : array-like
        Original observed data
    percentiles : list
        List of percentiles used
    n_simulations : int
        Number of Monte Carlo simulations
    random_state : int
        Random seed

    Returns:
    --------
    pd.DataFrame : Validation report with bias, RMSE, coverage
    dict : All synthetic percentile estimates
    """
    np.random.seed(random_state)

    n = len(data)
    fitted_dist = best_fit_info["best_fit"]["distribution"]

    # Store synthetic percentile estimates
    synthetic_estimates: dict[int | float, list[float]] = {p: [] for p in percentiles}

    print(f"\nRunning {n_simulations} Monte Carlo simulations...")

    for i in range(n_simulations):
        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{n_simulations} simulations")

        # Generate synthetic dataset from fitted distribution
        synthetic_data = fitted_dist.rvs(size=n, random_state=random_state + i)

        # Compute percentiles
        for p in percentiles:
            synthetic_estimates[p].append(np.percentile(synthetic_data, p))

    # Compute validation metrics
    validation_results: list[dict[str, str | float]] = []

    for i, p in enumerate(percentiles):
        bootstrap_value = bootstrap_table.iloc[i]["Value"]
        bootstrap_ci_lower = bootstrap_table.iloc[i]["CI_Lower"]
        bootstrap_ci_upper = bootstrap_table.iloc[i]["CI_Upper"]

        synthetic_values = np.array(synthetic_estimates[p])

        # Bias
        bias = np.mean(synthetic_values) - bootstrap_value

        # RMSE
        rmse = np.sqrt(np.mean((synthetic_values - bootstrap_value)**2))

        # Coverage: % of synthetic values within bootstrap CI
        coverage = np.mean((synthetic_values >= bootstrap_ci_lower) &
                          (synthetic_values <= bootstrap_ci_upper)) * 100

        # Relative bias (as percentage)
        relative_bias = (bias / bootstrap_value) * 100 if bootstrap_value != 0 else 0

        validation_results.append({
            "Percentile": f"{p}th",
            "Bootstrap_Value": bootstrap_value,
            "Synthetic_Mean": np.mean(synthetic_values),
            "Bias": bias,
            "Relative_Bias_%": relative_bias,
            "RMSE": rmse,
            "Coverage_%": coverage,
            "Synthetic_Std": np.std(synthetic_values),
        })

    validation_df = pd.DataFrame(validation_results)

    return validation_df, synthetic_estimates


def plot_validation(
    bootstrap_table: pd.DataFrame,
    validation_df: pd.DataFrame,
    synthetic_estimates: dict[int | float, list[float]],
    percentiles: list[int | float],
) -> None:
    """
    Create comprehensive validation plots.
    """
    n_percentiles = len(percentiles)

    # Plot 1: Bootstrap vs Synthetic percentiles with error bars
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot 1: Comparison plot
    ax: matplotlib.axes.Axes = axes[0, 0]
    x = np.arange(len(percentiles))
    width = 0.35

    bootstrap_vals = bootstrap_table["Value"].values
    synthetic_means = validation_df["Synthetic_Mean"].values
    bootstrap_errors = [(bootstrap_table["Value"] - bootstrap_table["CI_Lower"]).values,
                       (bootstrap_table["CI_Upper"] - bootstrap_table["Value"]).values]

    ax.bar(x - width/2, bootstrap_vals, width, label="Bootstrap", alpha=0.7)
    ax.bar(x + width/2, synthetic_means, width, label="Synthetic (Mean)", alpha=0.7)
    ax.errorbar(x - width/2, bootstrap_vals, yerr=bootstrap_errors, fmt="none",
                color="black", capsize=5, alpha=0.5)

    ax.set_xlabel("Percentile")
    ax.set_ylabel("Value")
    ax.set_title("Bootstrap vs Synthetic Percentiles")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}th" for p in percentiles])
    ax.legend()
    ax.grid(alpha=0.3)

    # Subplot 2: Bias plot
    ax = axes[0, 1]
    ax.bar(x, validation_df["Bias"].values, alpha=0.7, color="coral")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Bias")
    ax.set_title("Bias (Synthetic - Bootstrap)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}th" for p in percentiles])
    ax.grid(alpha=0.3)

    # Subplot 3: RMSE and Coverage
    ax = axes[1, 0]
    ax2 = ax.twinx()

    p1 = ax.bar(x - width/2, validation_df["RMSE"].values, width,
                label="RMSE", alpha=0.7, color="steelblue")
    p2 = ax2.plot(x, validation_df["Coverage_%"].values, "ro-",
                  linewidth=2, markersize=8, label="Coverage %")
    ax2.axhline(y=95, color="green", linestyle="--", linewidth=1.5,
                label="Target 95%", alpha=0.7)

    ax.set_xlabel("Percentile")
    ax.set_ylabel("RMSE", color="steelblue")
    ax2.set_ylabel("Coverage (%)", color="red")
    ax.set_title("RMSE and Coverage Rate")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}th" for p in percentiles])
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="red")
    ax.grid(alpha=0.3)

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines2, labels2, loc="upper left")

    # Subplot 4: Distribution of synthetic estimates for selected percentile
    ax = axes[1, 1]
    mid_percentile_idx = len(percentiles) // 2
    mid_percentile = percentiles[mid_percentile_idx]

    synthetic_vals = synthetic_estimates[mid_percentile]
    bootstrap_val = bootstrap_table.iloc[mid_percentile_idx]["Value"]
    ci_lower = bootstrap_table.iloc[mid_percentile_idx]["CI_Lower"]
    ci_upper = bootstrap_table.iloc[mid_percentile_idx]["CI_Upper"]

    ax.hist(synthetic_vals, bins=50, alpha=0.6, edgecolor="black",
            label="Synthetic estimates")
    ax.axvline(bootstrap_val, color="red", linestyle="--", linewidth=2,
               label="Bootstrap estimate")
    ax.axvline(ci_lower, color="blue", linestyle=":", linewidth=1.5,
               label="Bootstrap 95% CI")
    ax.axvline(ci_upper, color="blue", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Synthetic Distribution for {mid_percentile}th Percentile")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def complete_analysis(
    data: ArrayLike,
    percentiles: list[int | float] = [5, 10, 25, 50, 75, 90, 95],
    data_type: Literal["continuous", "count"] = "continuous",
    n_bootstrap: int = 10000,
    n_simulations: int = 1000,
    random_state: int = 42,
) -> dict[str, pd.DataFrame | dict[Any, Any]]:
    """
    Run complete bootstrap + validation analysis.

    Parameters:
    -----------
    data : array-like
        Observed data
    percentiles : list
        Percentiles to compute
    data_type : str
        'continuous' or 'count'
    n_bootstrap : int
        Number of bootstrap replicates
    n_simulations : int
        Number of Monte Carlo simulations
    random_state : int
        Random seed

    Returns:
    --------
    dict : Complete results including tables, fits, and validation
    """
    print("="*70)
    print("COMPLETE NORMATIVE TABLE ANALYSIS")
    print("="*70)

    # Step 1: Bootstrap
    print("\nSTEP 1: Bootstrap Analysis")
    print("-" * 70)
    bootstrap_table, bootstrap_est = bootstrap_percentiles(
        data, percentiles, n_bootstrap, random_state=random_state,
    )

    print("\nBootstrap Normative Table:")
    print(bootstrap_table.to_string(index=False, float_format="%.2f"))

    # Step 2: Fit distributions
    print("\n" + "="*70)
    print("STEP 2: Distribution Fitting")
    print("-" * 70)
    fit_info = fit_distributions(data, data_type)

    # Step 3: Monte Carlo validation
    print("\n" + "="*70)
    print("STEP 3: Monte Carlo Validation")
    print("-" * 70)
    validation_df, synthetic_est = monte_carlo_validation(
        bootstrap_table, fit_info, data, percentiles,
        n_simulations, random_state,
    )

    print("\nValidation Report:")
    print(validation_df.to_string(index=False, float_format="%.3f"))

    # Step 4: Summary statistics
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"Mean Absolute Bias:     {np.mean(np.abs(validation_df['Bias'])):.3f}")
    print(f"Mean Relative Bias:     {np.mean(np.abs(validation_df['Relative_Bias_%'])):.2f}%")
    print(f"Mean RMSE:              {np.mean(validation_df['RMSE']):.3f}")
    print(f"Mean Coverage:          {np.mean(validation_df['Coverage_%']):.1f}%")
    print(f"Coverage Range:         [{validation_df['Coverage_%'].min():.1f}%, "
          f"{validation_df['Coverage_%'].max():.1f}%]")

    # Interpretation
    print("\nINTERPRETATION:")
    mean_coverage = np.mean(validation_df["Coverage_%"])
    if mean_coverage >= 93 and mean_coverage <= 97:
        print("✓ Coverage is excellent - bootstrap CIs are well-calibrated")
    elif mean_coverage >= 90:
        print("✓ Coverage is good - bootstrap CIs are reasonably calibrated")
    else:
        print("⚠ Coverage is suboptimal - consider model fit issues")

    mean_rel_bias = np.mean(np.abs(validation_df["Relative_Bias_%"]))
    if mean_rel_bias < 2:
        print("✓ Bias is minimal - bootstrap estimates are accurate")
    elif mean_rel_bias < 5:
        print("✓ Bias is acceptable - bootstrap estimates are reasonably accurate")
    else:
        print("⚠ Bias is notable - bootstrap may not fully capture population")

    print("="*70)

    # Step 5: Plots
    plot_validation(bootstrap_table, validation_df, synthetic_est, percentiles)

    return {
        "bootstrap_table": bootstrap_table,
        "bootstrap_estimates": bootstrap_est,
        "fit_info": fit_info,
        "validation_table": validation_df,
        "synthetic_estimates": synthetic_est,
    }
