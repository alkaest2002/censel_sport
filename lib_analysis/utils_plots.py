"""Plotting utilities for statistical analysis."""

from base64 import b64encode
import io
from typing import Any

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd

# Constants
BASE_FIGURE_SIZE = (10, 8)
BASE_ALPHA = 0.5
MIN_DATA_POINTS = 3
BASE_FONTSIZE = 16

def _validate_data_points(
        data: NDArray[np.integer[Any] | np.floating[Any]],
        plot_name: str,
    ) -> tuple[NDArray[np.integer[Any] | np.floating[Any]], int]:
    """
        Checks minimum number of data-points.

    Parameters:
    ------------
        data: NDArray
            Data to plot

        plot_name: str
            Name of the plot

    Returns:
    -----------
    Tuple: The filtered data array and its size.
    """

    # Raise error if number of observations is lower than 3
    if data.size < MIN_DATA_POINTS:
        raise ValueError(
            f"---> Unable to plot {plot_name}: data points are not sufficient.",
        )

    # Raise error if any datapoint are below 0
    if np.any(data < 0):
        raise ValueError("---> Unable to plot {plot_name}: data must be non-negative.")

    # Raise error if data contains no finite values
    finite_mask: NDArray[np.bool_] = np.isfinite(data)
    if not np.any(finite_mask):
        raise ValueError(f"---> Unable to plot {plot_name}: data must contain at least one finite value")

    # Filter out non-finite values with warning
    if not np.all(finite_mask):
        original_size = data.size
        data = data[finite_mask]
        n = data.size
        print(f"---> Warning: {original_size - n} non-finite values were removed from the data")

    return data, data.size

def figure_to_svg_string(fig: Figure) -> str:
    """Convert a matplotlib figure to an SVG string ready for file saving.

    Takes a matplotlib figure object and converts it to an SVG format string
    that can be directly saved to an .svg file. The figure is automatically
    closed after conversion to free memory.

    Parameters:
    ------------
        fig: Matplotlib figure object to convert.

    Returns:
    -----------
    SVG string content ready to be written to a file.
    """
    # Initialize an in-memory text buffer for SVG content
    buffer = io.BytesIO()

    # Save figure to the buffer in SVG format then close it
    fig.savefig(
        buffer,
        format="svg",
        bbox_inches="tight",
        transparent=True,
        pad_inches=0.05,
    )

    # Close figure to free memory
    plt.close(fig)

    # Get the SVG content as string and reset buffer position
    buffer.seek(0)

    # Encode the buffer contents to a base64 string
    base64_encoded_string = b64encode(buffer.getvalue()).decode()

    return f"data:image/svg+xml;base64,{base64_encoded_string}"

def plot_histogram_with_fitted_model(
        data: NDArray[np.integer[Any] | np.floating[Any]],
        model_name: str,
        model: Any,
        bins: int | str | None = None,
        density: bool = False,
    ) -> str:
    """
    Create a histogram of observed data overlaid with the fitted theoretical distribution.

    For discrete data, creates a bar plot of observed frequencies with theoretical PMF.
    For continuous data, creates a histogram with theoretical PDF overlay.

    Parameters:
    -----------
    data : NDArray
        Data to plot (integer for discrete, float for continuous)

    model_name : str
        Name of the fitted distribution for plot title

    model : Any
        Fitted statistical model with pdf()/pmf() method

    bins : int, str, or None, optional
        Number of histogram bins for continuous data or binning strategy.
        Ignored for discrete data. Default uses 'auto' for continuous data.

    density : bool, optional
        If True, normalize histogram to show density (default).
        If False, show raw counts.

    Returns:
    --------
    str: SVG string of the generated histogram with fitted model
    """
    # Raise error if data size is insufficient
    data, n = _validate_data_points(data, "Histogram with Fitted Model")

    # Humanize model name for display
    model_name = model_name.replace("_", " ").title()

    # Determine if data is discrete (integer) or continuous
    is_discrete = np.issubdtype(data.dtype, np.integer) or np.allclose(data, np.round(data))

    # Create the plot
    figure, ax = plt.subplots(figsize=BASE_FIGURE_SIZE)

    # Handle discrete data
    if is_discrete:

        # Convert as int
        data = data.astype(int)

        # Get unique values and their counts
        unique_values, counts = np.unique(data, return_counts=True)

        # Calculate frequencies (normalized if density=True)
        if density:
            frequencies = counts / n
            ylabel = "Probability"
            theoretical_label = model_name
        else:
            frequencies = counts
            ylabel = "Count"
            theoretical_label = model_name

        # Create bar plot for observed data
        bar_width = 0.8
        ax.bar(unique_values, frequencies, width=bar_width, alpha=BASE_ALPHA,
            color="k", linewidth=0.5,
            label="Observed data")

        # Plot theoretical PMF
        x_range = np.arange(max(0, np.min(unique_values) - 1),
            np.max(unique_values) + 2)

        try:
            theoretical_probs = np.array([model.pmf(k) for k in x_range])
            if not density:
                theoretical_probs *= n

            ax.plot(x_range, theoretical_probs, color="k", linewidth=2,
                marker="o", markersize=6, label=theoretical_label)

        except AttributeError as e:
            raise AttributeError("---> Model must have pmf() method for discrete data") from e

        # Set integer ticks on x-axis
        ax.set_xticks(x_range[::max(1, len(x_range)//20)])

    # Handle continuous data
    else:

        # Set default bins if not provided
        if bins is None:
            bins = "auto"

        # Create histogram for observed data
        _, bin_edges, _ = ax.hist(
            data,
            bins=bins,
            density=density,
            rwidth=0.9,
            alpha=BASE_ALPHA, color="k",
            edgecolor="black",
            linewidth=0.5,
            label="Observed data",
        )

        # Determine ylabel based on density setting
        ylabel = "Density" if density else "Count"

        # Plot theoretical PDF
        x_min, x_max = np.min(data), np.max(data)
        x_range_continuous = np.linspace(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min), 1000)

        try:
            theoretical_density = np.array([model.pdf(x) for x in x_range_continuous])
            if not density:
                # Scale by sample size and bin width for count comparison
                bin_width = bin_edges[1] - bin_edges[0]
                theoretical_density *= n * bin_width

            ax.plot(x_range_continuous, theoretical_density, color="k", linewidth=2, label=model_name)

        except AttributeError as e:
            raise AttributeError("---> Model must have pdf() method for continuous data") from e

    # Common formatting for both discrete and continuous
    ax.set_xlabel("Values", fontsize=BASE_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=BASE_FONTSIZE)
    ax.legend(fontsize=BASE_FONTSIZE - 2, frameon=False)
    ax.yaxis.set_ticks_position("both")

    return figure_to_svg_string(figure)

def plot_bootstrap_percentile_with_ci(
        percentile_data: list[dict[str, Any]],
    ) -> str:
    """
    Create a plot of bootstrap percentile estimates with confidence intervals.

    Shows percentile values as points with confidence interval bands, useful for
    visualizing the uncertainty in percentile estimates from bootstrap sampling.

    Parameters:
    -----------
    percentile_data : list[dict]
        List of dictionaries, each containing:
        - "percentile": percentile value (0-100)
        - "value": estimated percentile value
        - "ci_level": confidence level (e.g., 0.95)
        - "ci_lower": lower confidence bound
        - "ci_upper": upper confidence bound
        - "std_error": standard error (optional, for display)

    Returns:
    --------
    str: SVG string of the generated percentile plot with confidence intervals
    """
    # Extract data arrays
    try:
        percentiles = np.array([item["percentile"] for item in percentile_data])
        values = np.array([item["value"] for item in percentile_data])
        ci_lower = np.array([item["ci_lower"] for item in percentile_data])
        ci_upper = np.array([item["ci_upper"] for item in percentile_data])
        ci_levels = np.array([item["ci_level"] for item in percentile_data])

        # Optional: extract standard errors if available
        std_errors = None
        if all("std_error" in item for item in percentile_data):
            std_errors = np.array([item["std_error"] for item in percentile_data])

    except KeyError as e:
        raise KeyError(f"---> Missing required key in percentile data: {e}") from e

    # Sort data by percentile for proper plotting
    sort_idx = np.argsort(percentiles)
    percentiles = percentiles[sort_idx]
    values = values[sort_idx]
    ci_lower = ci_lower[sort_idx]
    ci_upper = ci_upper[sort_idx]
    ci_levels = ci_levels[sort_idx]
    if std_errors is not None:
        std_errors = std_errors[sort_idx]

    # Create the plot
    figure, ax = plt.subplots(figsize=BASE_FIGURE_SIZE)

    # Plot confidence interval bands
    # Use the most common CI level for labeling
    unique_ci_levels, counts = np.unique(ci_levels, return_counts=True)
    most_common_ci = unique_ci_levels[np.argmax(counts)]
    ci_percentage = int(most_common_ci * 100)

    ax.fill_between(percentiles, ci_lower, ci_upper,
        alpha=BASE_ALPHA, color="#DDDDDD",
        label=f"{ci_percentage}% Confidence Interval")

    # Plot percentile estimates as points connected by lines
    ax.plot(percentiles, values, color="k", linewidth=2,
        marker="o", markersize=6, markerfacecolor="white",
        markeredgecolor="k", markeredgewidth=2,
        label="Percentile Estimates")

    # Add confidence interval bounds as dotted lines for clarity
    ax.plot(percentiles, ci_lower, color="darkgray", linewidth=1,
        linestyle=":", label="CI Bounds")
    ax.plot(percentiles, ci_upper, color="darkgray", linewidth=1,
        linestyle=":")

    # Formatting
    ax.set_xlabel("Percentile", fontsize=BASE_FONTSIZE)
    ax.set_ylabel("Value", fontsize=BASE_FONTSIZE)
    ax.legend(fontsize=BASE_FONTSIZE-2, frameon=False)

    # Set reasonable x-axis limits and ticks
    ax.set_xlim(max(0, np.min(percentiles) - 5), min(100, np.max(percentiles) + 5))

    # Set x-axis ticks at meaningful percentile values
    if np.max(percentiles) - np.min(percentiles) > 50:
        tick_step = 10
    elif np.max(percentiles) - np.min(percentiles) > 25:
        tick_step = 5
    else:
        tick_step = max(1, int((np.max(percentiles) - np.min(percentiles)) / 10))

    x_ticks = np.arange(0, 101, tick_step)
    x_ticks = x_ticks[(x_ticks >= np.min(percentiles) - 5) & (x_ticks <= np.max(percentiles) + 5)]
    ax.set_xticks(x_ticks)
    ax.yaxis.set_ticks_position("both")

    return figure_to_svg_string(figure)

def plot_qq_plot(
        data: NDArray[np.integer[Any] | np.floating[Any]],
        model_name: str,
        model: Any,
    ) -> str:
    """
    Create a Q-Q (quantile-quantile) plot comparing sample data to a normal distribution.

    Parameters:
    -----------
    data : NDArray
        Data to plot

    fitted_model: dict
        Dictionary containing the fitted model

    Returns:
    -------
    str: SVG string of the generated Q-Q plot.
    """
    # Raise error data size is insufficient
    data, n = _validate_data_points(data, "Q-Q Plot")

    # Humanize model name for display
    model_name = model_name.replace("_", " ").title()

    # Sort data in ascending order
    y: NDArray[np.floating[Any]] = np.sort(data)

    # Generate probability points using i/(N+2) to avoid extreme quantiles
    prob_points_array = np.linspace(1/(n+2), n/(n+2), n)
    x: NDArray[np.floating[Any]] = np.array([model.ppf(p) for p in prob_points_array])

    # Create a new figure with specified size for better SVG output
    figure, ax = plt.subplots(figsize=BASE_FIGURE_SIZE)

    # Create the Q-Q scatter plot
    ax.scatter(x, y, alpha=BASE_ALPHA, c="white", edgecolors="k", linewidths=0.5, s=50, label="Data points")

    # Plot diagonal reference line
    data_min: float = float(np.min([x, y]))
    data_max: float = float(np.max([x, y]))
    diag: NDArray[np.floating[Any]] = np.linspace(data_min, data_max, 1000)
    ax.plot(diag, diag, color="k", linestyle="--", linewidth=2, label="Fit line")

    # Add labels and formatting
    ax.set_xlabel(f"{model_name} Quantiles", fontsize=BASE_FONTSIZE)
    ax.set_ylabel("Sample Quantiles", fontsize=BASE_FONTSIZE)
    ax.legend(fontsize=BASE_FONTSIZE-2, frameon=False)
    ax.yaxis.set_ticks_position("both")

    return figure_to_svg_string(figure)

def plot_hanging_rootogram(
        data: NDArray[np.integer[Any] | np.floating[Any]],
        model_name: str,
        model: Any,
        max_count: int | None = None,
    ) -> str:
    """
    Create a hanging rootogram for discrete discrete data.

    Bars hang from the theoretical distribution line. If theoretical distribution
    overestimates a count, the observed bar doesn't reach the x-axis. If it
    underestimates, the observed bar crosses the x-axis (extends below zero).

    Parameters:
    -----------
    data : NDArray[np.integer]
        Integer discrete data to analyze

    model_name : str
        Name of the fitted distribution for plot title

    model : Any
        Fitted statistical model with pmf() method for probability mass function

    max_count : int, optional
        Maximum count value to display. If None, uses max(data) + 2

    Returns:
    --------
    str: SVG string of the generated hanging rootogram
    """
    # Raise error data size is insufficient
    data, n = _validate_data_points(data, "Rootgram Plot")

    # Humanize model name for display
    model_name = model_name.replace("_", " ").title()

    # Validate integer data
    if not np.issubdtype(data.dtype, np.integer):
        if not np.allclose(data, np.round(data)):
            raise TypeError("---> Rootogram requires integer count data")
        data = data.astype(int)

    # Determine count range for analysis
    data_max: int = np.max(data)
    max_count = data_max + 2 if max_count is None else min(max_count, data_max + 2)

    # Create count range
    counts = np.arange(0, max_count + 1)

    # Calculate observed frequencies
    observed_freq = np.zeros(max_count + 1)
    unique_values, value_counts = np.unique(data, return_counts=True)

    # Fill observed frequencies for values within our range
    for val, count in zip(unique_values, value_counts, strict=False):
        if val <= max_count:
            observed_freq[val] = count

    # Calculate expected frequencies using the model
    try:
        expected_prob = np.array([model.pmf(k) for k in counts])
        expected_freq = n * expected_prob
    except AttributeError as e:
        raise AttributeError("---> Model must have pmf() method for probability mass function") from e

    # Handle potential numerical issues
    expected_freq = np.maximum(expected_freq, 1e-10)  # Avoid zero expected values

    # Calculate square root transformations
    observed_sqrt = np.sqrt(observed_freq)
    expected_sqrt = np.sqrt(expected_freq)

    # Create the plot
    figure, ax = plt.subplots(figsize=BASE_FIGURE_SIZE)

    # Create bars hanging from the theoretical distribution
    bar_width = 0.1

    # For each count, create a bar that hangs from expected_sqrt down to
    # expected_sqrt - observed_sqrt (which could be negative)
    bar_bottoms = expected_sqrt - observed_sqrt  # Where bars end
    bar_heights = observed_sqrt  # Height of each bar

    # Color bars based on whether they cross the x-axis (theoretical underestimates)
    # or don't reach it (theoretical overestimates)
    colors = ["red" if bottom < 0 else "steelblue" for bottom in bar_bottoms]

    # Plot hanging bars
    _ = ax.bar(counts, bar_heights, bottom=bar_bottoms,
                  width=bar_width, color=colors, alpha=BASE_ALPHA,
                  edgecolor="black", linewidth=0.5)

    # Plot theoretical (expected) square root line - this is where bars hang from
    ax.plot(counts, expected_sqrt, color="black", linewidth=2,
            marker="o", markersize=4, label=model_name)

    # Add 1 reference line (x-axis)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=BASE_ALPHA, linewidth=1.5)

    # Add 1 reference line (x-axis)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=BASE_ALPHA, linewidth=1.5,
               label="Reference line (x-axis)")

    # Add 1 reference line (x-axis)
    ax.axhline(y=-1, color="gray", linestyle="--", alpha=BASE_ALPHA, linewidth=1.5)

    # Formatting
    ax.set_xlabel("Count Values", fontsize=BASE_FONTSIZE)
    ax.set_ylabel("Square Root of Frequency", fontsize=BASE_FONTSIZE)
    ax.legend(fontsize=BASE_FONTSIZE-2, frameon=False)

    # Set integer ticks on x-axis
    ax.set_xticks(counts[::max(1, len(counts)//10)])  # Show reasonable number of ticks
    ax.yaxis.set_ticks_position("both")

    return figure_to_svg_string(figure)

def plot_montecarlo(comparison_data: list[dict[str, Any]]) -> str:
    """
    Create a scatter plot comparing bootstrap and Monte Carlo percentile estimates.

    Shows bootstrap values vs Monte Carlo values with Monte Carlo IQR as error bars.
    Points should ideally lie on the main diagonal, indicating agreement between methods.

    Parameters:
    -----------
    comparison_data : list[dict]
        List of dictionaries, each containing:
        - "percentile": percentile identifier (can be string or number)
        - "bootstrap_value": bootstrap estimate
        - "bootstrap_ci_lower": bootstrap CI lower bound
        - "bootstrap_ci_upper": bootstrap CI upper bound
        - "montecarlo_value": Monte Carlo estimate
        - "montecarlo_std": Monte Carlo standard deviation
        - "montecarlo_min": Monte Carlo minimum value
        - "montecarlo_max": Monte Carlo maximum value
        - "montecarlo_iqr": Monte Carlo interquartile range
        - "bias": difference (montecarlo - bootstrap)
        - "relative_bias_%": relative bias percentage
        - "rmse": root mean square error
        - "coverage_%": coverage percentage

    Returns:
    --------
    str: SVG string of the generated Monte Carlo comparison plot
    """
    if len(comparison_data) < MIN_DATA_POINTS:
        raise ValueError(
            f"---> Unable to plot Monte Carlo comparison: at least {MIN_DATA_POINTS} points required",
        )

    # Convert data into DataFrame
    data: pd.DataFrame = pd.DataFrame(comparison_data)

    # Create the plot
    figure, ax = plt.subplots(figsize=BASE_FIGURE_SIZE)

    # Add perfect agreement diagonal line
    data_min = data.loc[:, ["bootstrap_value", "montecarlo_value"]].min().min()
    data_max = data.loc[:, ["bootstrap_value", "montecarlo_value"]].max().max()

    # Extend the diagonal line slightly beyond data range
    margin = 0.05 * (data_max - data_min)
    diag_min = data_min - margin
    diag_max = data_max + margin

    diagonal = np.linspace(diag_min, diag_max, 100)
    ax.plot(diagonal, diagonal, c="#CCCCCC", linestyle="--", linewidth=2,
        label="Perfect Agreement")

    # Create scatter plot with Monte Carlo IQR as error bars
    _ = ax.errorbar(
        data["bootstrap_value"], data["montecarlo_value"],
        yerr=data["montecarlo_iqr"].mul(1.5).div(2),  # IQR * 1.5 divided by 2 for symmetric error bars
        fmt="o", markersize=6, markerfacecolor="white",
        markeredgecolor="k", markeredgewidth=2,
        ecolor="k", elinewidth=1.5, capsize=0, capthick=0,
        label="Monte Carlo values, error-bars = IQR * 1.5",
    )

    # Add percentile labels to points
    for _, row in data.iterrows():
        ax.annotate(f"{row['percentile']}",
            (row["bootstrap_value"], row["montecarlo_value"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=BASE_FONTSIZE-2,
            alpha=0.8,
            ha="left",
            va="bottom")

    # Formatting
    ax.set_xlabel("Bootstrap Values", fontsize=BASE_FONTSIZE)
    ax.set_ylabel("Monte Carlo Values", fontsize=BASE_FONTSIZE)
    ax.legend(fontsize=BASE_FONTSIZE-2, frameon=False)

    # Set axis limits with some padding
    ax.set_xlim(diag_min, diag_max)
    ax.set_ylim(diag_min, diag_max)
    ax.yaxis.set_ticks_position("both")

    return figure_to_svg_string(figure)
