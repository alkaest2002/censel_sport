"""Plotting utilities for statistical analysis."""

from base64 import b64encode
import io
from typing import Any

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd

# Constants
MIN_DATA_POINTS = 3
BASE_FIGURE_SIZE = (10, 8)
BASE_ALPHA = 0.5
BASE_FONTSIZE = 16
PRIMARY_COLOR = "#5580B0"
SECONDARY_COLOR = "#E23122"
NEUTRAL_COLOR = "#888888"
NEUTRAL_COLOR_LIGHT = "#CCCCCC"


def _validate_data_points(
        data: NDArray[np.number[Any]],
        plot_name: str,
    ) -> tuple[NDArray[np.number[Any]], int]:
    """
        Validate the data points for plotting.

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

def _get_x_lim_with_padding(
        data: NDArray[np.number[Any]],
    ) -> tuple[float | int, float | int]:
    """Compute data min and max limits with padding.

    Parameters:
    ------------
    data: NDArray
        Numpy array containing data.

    Returns:
    -----------
    tuple: Data min and max with padding.
    """

    x_min: float | int = max(0, np.min(data))
    x_max: float | int = np.max(data)
    min_padding: float | int = max(0, (x_max - x_min) * 0.1)
    max_padding: float | int = (x_max - x_min) * 0.1
    return x_min - min_padding, x_max + max_padding

def figure_to_svg_string(fig: Figure) -> str:
    """Convert a matplotlib figure to an SVG string.

    Parameters:
    ------------
    fig: Figure
        Matplotlib figure object to convert.

    Returns:
    -----------
    str: SVG content ready to be written to a file.
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
        data: NDArray[np.number[Any]],
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
            frequencies: NDArray[np.floating[Any]] = counts / n
            ylabel: str = "Probabilità"
        else:
            frequencies = counts
            ylabel = "Frequenza"

        # Create bar plot for observed data
        ax.bar(unique_values, frequencies, width=0.2, color=NEUTRAL_COLOR, linewidth=1, label="Dati osservati")

        # Create x range for theoretical PMF
        x_min, x_max = max(0, np.min(unique_values)), np.max(unique_values)
        x: NDArray[np.floating[Any]] = np.arange(x_min, x_max + 1)

        try:
            # Calculate theoretical PMF
            theoretical_probs = np.array([model.pmf(k) for k in x])

            # Scale by sample size if showing counts
            if not density:
                theoretical_probs *= n

            # Plot theoretical PMF
            ax.plot(x, theoretical_probs, color=PRIMARY_COLOR, linewidth=2, marker="o", markersize=4, label=model_name)

        except AttributeError as e:
            raise AttributeError("---> Model must have pmf() method for discrete data") from e

    # Handle continuous data
    else:

        # Set default bins if not provided
        bins = bins or "auto"

        # Create histogram for observed data
        _, bin_edges, _ = ax.hist(
            data,
            bins=bins,
            density=density,
            rwidth=0.9,
            color=NEUTRAL_COLOR_LIGHT,
            edgecolor=NEUTRAL_COLOR_LIGHT,
            linewidth=0.5,
            label="Dati osservati",
        )

        # Determine ylabel based on density setting
        ylabel = "Densità" if density else "Frequenza"

        # Compute x range for theoretical PDF
        x_min, x_max = max(0, np.min(data)), np.max(data)
        x = np.linspace(x_min, x_max, 1000)

        try:
            # Calculate theoretical PDF
            theoretical_density = np.array([model.pdf(x_point) for x_point in x])

            # Scale by sample size and bin width if showing counts
            if not density:
                bin_width = bin_edges[1] - bin_edges[0]
                theoretical_density *= n * bin_width

            # Plot theoretical PDF
            ax.plot(x, theoretical_density, color=PRIMARY_COLOR, linewidth=2, label=model_name)

        except AttributeError as e:
            raise AttributeError("---> Model must have pdf() method for continuous data") from e

    # Common formatting for both discrete and continuous
    ax.set_xlabel("Valori", fontsize=BASE_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=BASE_FONTSIZE)
    ax.legend(fontsize=BASE_FONTSIZE - 2, frameon=False)
    ax.yaxis.set_ticks_position("both")
    ax.set_xlim(*_get_x_lim_with_padding(data))

    return figure_to_svg_string(figure)

def plot_qq_plot(
        data: NDArray[np.number[Any]],
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
    ax.scatter(x, y, alpha=BASE_ALPHA, c=PRIMARY_COLOR,
               edgecolors=PRIMARY_COLOR, linewidths=0.5, s=50, label="Dati osservati")

    # Plot diagonal reference line
    data_min: float = float(np.min([x, y]))
    data_max: float = float(np.max([x, y]))
    diag: NDArray[np.floating[Any]] = np.linspace(data_min, data_max, 1000)
    ax.plot(diag, diag, color="k", linestyle="--", linewidth=2, label="Linea di riferimento (y=x)")

    # Add labels and formatting
    ax.set_xlabel(f"Quantili {model_name}", fontsize=BASE_FONTSIZE)
    ax.set_ylabel("Quantili del campione", fontsize=BASE_FONTSIZE)
    ax.legend(fontsize=BASE_FONTSIZE-2, frameon=False)
    ax.yaxis.set_ticks_position("both")
    ax.set_xlim(*_get_x_lim_with_padding(data))

    return figure_to_svg_string(figure)

def plot_hanging_rootogram(
        data: NDArray[np.number[Any]],
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

    # For each count, create a bar that hangs from expected_sqrt down to
    # expected_sqrt - observed_sqrt (which could be negative)
    bar_bottoms = expected_sqrt - observed_sqrt  # Where bars end
    bar_heights = observed_sqrt  # Height of each bar

    # Plot hanging bars
    _ = ax.bar(counts, bar_heights, bottom=bar_bottoms, width=0.2, color=NEUTRAL_COLOR, linewidth=1)

    # Plot theoretical (expected) square root line - this is where bars hang from
    ax.plot(counts, expected_sqrt, color=PRIMARY_COLOR, linewidth=2, marker="o", markersize=4, label=model_name)

    # Add 1 reference line (x-axis)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=BASE_ALPHA, linewidth=1.5)

    # Add 1 reference line (x-axis)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=BASE_ALPHA, linewidth=1.5, label="Linea di riferimento (asse x)")

    # Add 1 reference line (x-axis)
    ax.axhline(y=-1, color="gray", linestyle="--", alpha=BASE_ALPHA, linewidth=1.5)

    # Formatting
    ax.set_xlabel("Valori", fontsize=BASE_FONTSIZE)
    ax.set_ylabel("Radice quadrata della frequenza", fontsize=BASE_FONTSIZE)
    ax.legend(fontsize=BASE_FONTSIZE-2, frameon=False)

    # Set integer ticks on x-axis
    ax.set_xlim(*_get_x_lim_with_padding(data))
    ax.yaxis.set_ticks_position("both")

    return figure_to_svg_string(figure)

def plot_bootstrap_percentile_with_ci(
        bootstrap_requested_percentiles: list[dict[str, Any]],
        bootstrap_all_percentiles: list[dict[str, Any]],
    ) -> str:
    """
    Create a plot of bootstrap percentile estimates with confidence intervals.

    Shows percentile values as points with confidence interval bands, useful for
    visualizing the uncertainty in percentile estimates from bootstrap sampling.

    Parameters:
    -----------
    bootstrap_requested_percentiles : list[dict]
        List of requested percentiles

    bootstrap_all_percentiles: list[dict]
        List of all percentiles from 1 to 99

    Returns:
    --------
    str: SVG string of the generated percentile plot with confidence intervals
    """
    # Extract data arrays
    all_df = pd.DataFrame(bootstrap_all_percentiles)
    requested_df = pd.DataFrame(bootstrap_requested_percentiles)
    ci_level: float = bootstrap_all_percentiles[0]["ci_level"]


    # Create the plot
    figure, ax = plt.subplots(figsize=BASE_FIGURE_SIZE)

    ax.fill_between(
        x=all_df["percentile"],
        y1=all_df["ci_lower"],
        y2=all_df["ci_upper"],
        alpha=BASE_ALPHA, color="lightblue",
        label=f"Intervallo di Confidenza {ci_level * 100}%",
    )

    # Plot percentile estimates as points connected by lines
    ax.plot(requested_df["percentile"], requested_df["value"], color=PRIMARY_COLOR, linewidth=0,
        marker="o", markersize=6, markerfacecolor=PRIMARY_COLOR,
        markeredgecolor=PRIMARY_COLOR, markeredgewidth=2,
        label="Stime Percentili")

    # plot a vertical line for each requested percentile
    for rp, rv in zip(requested_df["percentile"], requested_df["value"], strict=True):
        ax.vlines(rp, ymin=0, ymax=rv, colors=PRIMARY_COLOR, linewidth=1)

    # Formatting
    ax.set_xlabel("Percentili", fontsize=BASE_FONTSIZE)
    ax.set_ylabel("Valori", fontsize=BASE_FONTSIZE)
    ax.legend(fontsize=BASE_FONTSIZE-2, frameon=False)

    # Constrain y axis to start at 0
    ax.set_ylim(bottom=0)

    # Set reasonable x-axis limits and ticks
    ax.set_xlim(max(0, requested_df["percentile"].min() - 5), min(100, requested_df["percentile"].max() + 5))
    ax.set_xticks(requested_df["percentile"])
    ax.yaxis.set_ticks_position("both")

    return figure_to_svg_string(figure)

def plot_montecarlo_vs_bootstrap(
        bootstrap_percentiles: list[dict[str, Any]],
        montecarlo_percentiles: list[dict[str, Any]],
    ) -> str:
    """
    Create a scatter plot comparing bootstrap and Monte Carlo percentile estimates.

    Shows bootstrap values vs Monte Carlo values with Monte Carlo IQR as error bars.
    Points should ideally lie on the main diagonal, indicating agreement between methods.

    Parameters:
    -----------
    bootstrap_percentiles : list[dict]
        List of dictionaries, each containing:
        - "percentile": percentile value (0-100)
        - "value": bootstrap percentile estimate
        - "first_quartile": first quartile of Monte Carlo IQR
        - "third_quartile": third quartile of Monte Carlo IQR
    montecarlo_percentiles : list[dict]
        List of dictionaries, each containing:
        - "percentile": percentile value (0-100)
        - "montecarlo_value": Monte Carlo percentile estimate

    Returns:
    --------
    str: SVG string of the generated Monte Carlo comparison plot
    """
    # Create figure following utils_plots conventions
    figure, ax = plt.subplots(figsize=BASE_FIGURE_SIZE)

    # Prepare data
    percentile_labels = []
    positions = []
    box_width = 0.1
    offset = 0.1  # Distance between paired boxes

    for i, bootstrap_perc in enumerate(bootstrap_percentiles):
        percentile = bootstrap_perc["percentile"]
        percentile_labels.append(f"{percentile}")

        # Center position for this percentile group
        center_position = i + 1
        positions.append(center_position)

        # Positions for bootstrap (left) and Monte Carlo (right) boxes
        bootstrap_pos = center_position - offset
        montecarlo_pos = center_position + offset

        # === BOOTSTRAP BOX PLOT ===
        bootstrap_q1 = bootstrap_perc["first_quartile"]
        bootstrap_q3 = bootstrap_perc["third_quartile"]
        bootstrap_min = bootstrap_perc["min"]
        bootstrap_max = bootstrap_perc["max"]

        # Draw bootstrap box
        bootstrap_box_height = bootstrap_q3 - bootstrap_q1
        bootstrap_box = plt.Rectangle(
            (bootstrap_pos - box_width/2, bootstrap_q1), box_width, bootstrap_box_height,
            facecolor=PRIMARY_COLOR, edgecolor=PRIMARY_COLOR, linewidth=1.5,
        )
        ax.add_patch(bootstrap_box)

        # Draw bootstrap whiskers
        ax.vlines(bootstrap_pos, bootstrap_q1,
                  bootstrap_min, colors=PRIMARY_COLOR, linewidth=1.5)
        ax.hlines(bootstrap_min, bootstrap_pos - box_width/2,
                  bootstrap_pos + box_width/2, colors=PRIMARY_COLOR, linewidth=1.5)
        ax.vlines(bootstrap_pos, bootstrap_q3, bootstrap_max, colors=PRIMARY_COLOR, linewidth=1.5)
        ax.hlines(bootstrap_max, bootstrap_pos - box_width/2,
                  bootstrap_pos + box_width/2, colors=PRIMARY_COLOR, linewidth=1.5)

        # === MONTE CARLO BOX PLOT ===
        # Get corresponding Monte Carlo data
        mc_data = montecarlo_percentiles[i]
        mc_q1 = mc_data["first_quartile"]
        mc_q3 =mc_data["third_quartile"]
        mc_min = mc_data["min"]
        mc_max = mc_data["max"]

        # Draw Monte Carlo box
        mc_box_height = mc_q3 - mc_q1
        mc_box = plt.Rectangle(
            (montecarlo_pos - box_width/2, mc_q1),
            box_width, mc_box_height,
            facecolor=SECONDARY_COLOR, edgecolor=SECONDARY_COLOR, linewidth=1.5,
        )
        ax.add_patch(mc_box)

        # Draw Monte Carlo whiskers
        ax.vlines(montecarlo_pos, mc_q1, mc_min, colors=SECONDARY_COLOR, linewidth=1.5)
        ax.hlines(mc_min, montecarlo_pos - box_width/2, montecarlo_pos + box_width/2,
                    colors=SECONDARY_COLOR, linewidth=1.5)
        ax.vlines(montecarlo_pos, mc_q3, mc_max, colors=SECONDARY_COLOR, linewidth=1.5)
        ax.hlines(mc_max, montecarlo_pos - box_width/2, montecarlo_pos + box_width/2,
                    colors=SECONDARY_COLOR, linewidth=1.5)

    # Formatting following utils_plots conventions
    ax.set_xlim(0.5, len(positions) + 0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(percentile_labels)
    ax.set_xlabel("Percentili", fontsize=BASE_FONTSIZE)
    ax.set_ylabel("Valori", fontsize=BASE_FONTSIZE)

    # Position y-axis ticks on both sides following utils_plots
    ax.yaxis.set_ticks_position("both")

    # Create legend following utils_plots style (frameon=False)
    legend_elements = [
        Line2D([0], [0], color=PRIMARY_COLOR, marker="s", linestyle="None",
               markersize=10, markerfacecolor=PRIMARY_COLOR, markeredgecolor=PRIMARY_COLOR,
               label="Distribuzione Bootstrap"),
        Line2D([0], [0], color=SECONDARY_COLOR, marker="s", linestyle="None",
               markersize=10, markerfacecolor=SECONDARY_COLOR, markeredgecolor=SECONDARY_COLOR,
               label="Distribuzione Monte Carlo"),
    ]
    ax.legend(handles=legend_elements, fontsize=BASE_FONTSIZE-2, frameon=False, loc="upper left")

    return figure_to_svg_string(figure)
