"""
Plotting utilities for evaluation.

Provides consistent styling and helper functions for all evaluation plots.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Global Style Configuration
PLOT_STYLE = {
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 14,
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def apply_style() -> None:
    """Apply consistent plotting style globally."""
    plt.rcParams.update(PLOT_STYLE)


def save_figure(fig: plt.Figure, directory: Path, name: str, dpi: int = 150) -> Path:
    """
    Save a matplotlib figure to the specified directory.
    
    Args:
        fig: Matplotlib figure to save.
        directory: Directory to save the figure in.
        name: Filename (without extension).
        dpi: Resolution (default 150).
        
    Returns:
        Path to the saved file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    
    filepath = directory / f"{name}.png"
    fig.savefig(filepath, bbox_inches="tight", dpi=dpi, facecolor="white")
    plt.close(fig)
    
    logger.info(f"Saved plot: {filepath}")
    return filepath


def create_figure(
    nrows: int = 1, 
    ncols: int = 1, 
    figsize: Optional[tuple] = None
) -> tuple:
    """
    Create a figure with consistent styling.
    
    Args:
        nrows: Number of subplot rows.
        ncols: Number of subplot columns.
        figsize: Optional figure size tuple.
        
    Returns:
        Tuple of (figure, axes).
    """
    apply_style()
    
    if figsize is None:
        figsize = (8 * ncols, 5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, axes
