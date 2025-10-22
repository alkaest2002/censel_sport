"""Plotting utilities for statistical analysis."""

import io
from typing import TYPE_CHECKING, Any, Literal

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from lib.utils_distributions import DistributionType, FitFunctionType, get_distributions
from lib.utils_generic import is_falsy

if TYPE_CHECKING:
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

def qq_plot(data_dict: dict[str, Any]) -> str:
    """
    Create a Q-Q (quantile-quantile) plot comparing sample data to a normal distribution.

    This function generates a Q-Q plot to assess whether the given data follows a normal
    distribution. Points that fall approximately along the diagonal line suggest the data
    is normally distributed. The plot can be saved as an SVG file and/or displayed.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing data

    Returns:
    -------
    str: SVG string of the generated Q-Q plot.
    """
    # Extract from dictionary
    metric_config: dict[str, Any] = data_dict.get("metric_config", {})
    metric_type: Literal["time", "count"] | None = metric_config.get("metric_type")
    clean: dict[str, Any] = data_dict.get("clean", {})
    fit: dict[str, Any] = data_dict.get("fit", {})
    data: NDArray[np.integer[Any] | np.floating[Any]] = clean.get("data", np.array([]))
    best_model: dict[str, Any] = fit.get("best_model", {})

    # Raise error if something is missing
    if any(map(is_falsy, (clean, data))):
        raise ValueError("The data dictionary does not contain all required parts.")

    # Raise error if number of observations is lower than 3
    if n := data.size < MIN_DATA_POINTS:
        raise ValueError(f"Cannot create meaningful Q-Q plot: need at least 3 data points, got {data.size}")


    # Get distributions
    distributions: dict[str, tuple[DistributionType, FitFunctionType]] =\
        get_distributions(metric_type, best_model["name"])

    # Get best model class
    model_class, _ = distributions[best_model["name"]]

    # Instantiate best model class with fitted params
    try:
        model = model_class(*best_model["params"])
    except (TypeError, ValueError) as e:
        raise ValueError(f"Failed to instantiate model {best_model['name']}: {e}") from e


    # Check for finite values
    finite_mask: NDArray[np.bool_] = np.isfinite(data)
    if not np.any(finite_mask):
        raise ValueError("Unable to compute Q-Q plot: data must contain at least one finite value")

    # Use only finite values and warn if some were removed
    if not np.all(finite_mask):
        data = data[finite_mask]
        print(f"Warning: {np.sum(~finite_mask)} non-finite values were removed from the data")

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
    ax.set_title(f"Q-Q Plot: sample vs {best_model['name']} distribution", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    return figure_to_svg_string(figure)
