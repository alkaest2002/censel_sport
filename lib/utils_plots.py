"""Plotting utilities for statistical analysis."""

import io
from typing import Any

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# Constants
DEFAULT_FIGURE_SIZE = (8, 8)
DEFAULT_PAD_INCHES = 0.05
MIN_DATA_POINTS = 3

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
    buffer = io.StringIO()

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
    svg_string = buffer.getvalue()

    # Close the buffer
    buffer.close()

    return svg_string

def plot_qq_plot(
        data: NDArray[np.integer[Any] | np.floating[Any]],
        model_name: str,
        model: Any,
    ) -> str:
    """
    Create a Q-Q (quantile-quantile) plot comparing sample data to a normal distribution.

    This function generates a Q-Q plot to assess whether the given data follows a normal
    distribution. Points that fall approximately along the diagonal line suggest the data
    is normally distributed. The plot can be saved as an SVG file and/or displayed.

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
    # Get the size of the data first
    n = data.size

    # Raise error if number of observations is lower than 3
    if n < MIN_DATA_POINTS:
        raise ValueError(f"---> Cannot create meaningful Q-Q plot: need at least 3 data points, got {n}")

    # Check for finite values
    finite_mask: NDArray[np.bool_] = np.isfinite(data)
    if not np.any(finite_mask):
        raise ValueError("---> Unable to compute Q-Q plot: data must contain at least one finite value")

    # Use only finite values and warn if some were removed
    if not np.all(finite_mask):
        data = data[finite_mask]
        n = data.size  # Update n after filtering
        print(f"---> Warning: {np.sum(~finite_mask)} non-finite values were removed from the data")

    # Sort data in ascending order
    y: NDArray[np.floating[Any]] = np.sort(data)

    # Generate probability points using i/(N+2) to avoid extreme quantiles
    prob_points_array = np.linspace(1/(n+2), n/(n+2), n)
    x: NDArray[np.floating[Any]] = np.array([model.ppf(p) for p in prob_points_array])

    # Create a new figure with specified size for better SVG output
    figure, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)

    # Create the Q-Q scatter plot
    ax.scatter(x, y, alpha=.5, edgecolors="black", linewidths=0.5, s=50, label="Data points")

    # Plot diagonal reference line
    data_min: float = float(np.min([x, y]))
    data_max: float = float(np.max([x, y]))
    diag: NDArray[np.floating[Any]] = np.linspace(data_min, data_max, 1000)
    ax.plot(diag, diag, color="red", linestyle="--", linewidth=2, label="Perfect fit line")

    # Set equal aspect ratio for better interpretation
    ax.set_aspect("equal")

    # Add labels and formatting
    ax.set_xlabel("Theoretical Quantiles", fontsize=12)
    ax.set_ylabel("Sample Quantiles", fontsize=12)
    ax.set_title(f"Q-Q Plot: sample vs {model_name} distribution", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    return figure_to_svg_string(figure)

def plot_hanging_rootogram(
        data: NDArray[np.integer[Any]],
        model_name: str,
        model: Any,
        max_count: int | None = None,
    ) -> str:
    """
    Create a hanging rootogram for discrete count data.

    A hanging rootogram compares observed frequencies (square roots) to expected
    frequencies from a fitted discrete distribution. Bars "hang" from the expected
    values, with deviations from zero indicating poor fit. This visualization is
    particularly useful for count data (Poisson, Negative Binomial, Zero-Inflated, etc.).

    Parameters:
    -----------
    data : NDArray[np.integer]
        Integer count data to analyze

    model_name : str
        Name of the fitted distribution for plot title

    model : Any
        Fitted statistical model with pmf() method for probability mass function

    max_count : int, optional
        Maximum count value to display. If None, uses max(data) + 2

    Returns:
    --------
    str
        SVG string of the generated hanging rootogram
    """
    # Validate minimum data requirements
    n = data.size
    if n < MIN_DATA_POINTS:
        raise ValueError(
            f"---> Cannot create meaningful rootogram: need at least {MIN_DATA_POINTS} data points, got {n}",
        )

    # Check for finite integer values
    finite_mask: NDArray[np.bool_] = np.isfinite(data)
    if not np.any(finite_mask):
        raise ValueError("---> Unable to compute rootogram: data must contain at least one finite value")

    # Filter out non-finite values with warning
    if not np.all(finite_mask):
        original_size = data.size
        data = data[finite_mask]
        n = data.size
        print(f"---> Warning: {original_size - n} non-finite values were removed from the data")

    # Validate integer data
    if not np.issubdtype(data.dtype, np.integer):
        if not np.allclose(data, np.round(data)):
            raise TypeError("---> Rootogram requires integer count data")
        data = data.astype(int)

    # Ensure non-negative counts
    if np.any(data < 0):
        raise ValueError("---> Count data must be non-negative")

    # Determine count range for analysis
    data_max = int(np.max(data))
    max_count = min(data_max + 2, 50) if max_count is None else min(max_count, data_max + 2)

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

    # Calculate hanging values (deviations from expected)
    hanging_values = observed_sqrt - expected_sqrt

    # Create the plot
    figure, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)

    # Create bars hanging from the expected line
    bar_width = 0.8
    colors = ["red" if h < 0 else "steelblue" for h in hanging_values]

    # Plot hanging bars
    _ = ax.bar(counts, hanging_values, bottom=expected_sqrt,
                  width=bar_width, color=colors, alpha=0.7,
                  edgecolor="black", linewidth=0.5)

    # Plot expected square root line
    ax.plot(counts, expected_sqrt, color="black", linewidth=2,
            marker="o", markersize=4, label=f"Expected √freq ({model_name})")

    # Add zero reference line for hanging values
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5,
               label="Perfect fit line")

    # Formatting
    ax.set_xlabel("Count Values", fontsize=12)
    ax.set_ylabel("Square Root of Frequency", fontsize=12)
    ax.set_title(f"Hanging Rootogram: {model_name} Distribution Fit",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=11)

    # Set integer ticks on x-axis
    ax.set_xticks(counts[::max(1, len(counts)//20)])  # Show reasonable number of ticks

    # Add text box with interpretation guide
    textstr = "Bars above expected line: overestimated\nBars below expected line: underestimated"
    props = { "boxstyle": "round", "facecolor": "wheat", "alpha": 0.8 }
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=props)

    # Adjust layout to prevent clipping
    plt.tight_layout()

    return figure_to_svg_string(figure)

