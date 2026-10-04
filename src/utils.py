"""
Utility functions for evaluation metrics, plotting, and reproducibility.
"""

import os
import random
from typing import Dict, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    """
    Set seeds for torch, numpy, and random to guarantee deterministic results.

    Parameters
    ----------
    seed : int, default=42
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(directory_path: str) -> str:
    """
    Ensure a directory exists, creating parent directories if necessary.

    Parameters
    ----------
    directory_path : str
        Target directory path.

    Returns
    -------
    str
        Validated directory path.
    """
    os.makedirs(directory_path, exist_ok=True)
    return directory_path


def compute_metrics(pred: np.ndarray, exact: np.ndarray) -> Dict[str, float]:
    """
    Compute rigorous mathematical error metrics between predicted and exact solutions.

    Parameters
    ----------
    pred : np.ndarray
        Array of predicted values from the neural network.
    exact : np.ndarray
        Array of ground truth analytical solution values.

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'max_abs_error': L_infinity norm, max |pred - exact|
        - 'mean_squared_error': MSE
        - 'root_mean_squared_error': RMSE
        - 'relative_l2_error': ||pred - exact||_2 / ||exact||_2
    """
    diff = np.abs(pred - exact)
    max_abs_error = float(np.max(diff))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))

    norm_exact = np.linalg.norm(exact)
    if norm_exact > 0:
        relative_l2 = float(np.linalg.norm(pred - exact) / norm_exact)
    else:
        relative_l2 = float("nan")

    return {
        "max_abs_error": max_abs_error,
        "mean_squared_error": mse,
        "root_mean_squared_error": rmse,
        "relative_l2_error": relative_l2,
    }


def setup_figure_style() -> None:
    """Configure matplotlib settings for clean, publication-quality figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 14,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.5,
        "grid.linestyle": ":",
    })


def plot_solution_comparison(
    x: np.ndarray,
    y_exact: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_filename: Optional[str] = None,
    xlabel: str = "x",
    ylabel: str = r"$\Psi(x)$",
    pred_label: str = "Neural Network Solution",
    exact_label: str = "Analytical Solution",
    show: bool = True,
) -> None:
    """
    Plot comparison between analytical and neural network solutions with error subplot.

    Parameters
    ----------
    x : np.ndarray
        Spatial or temporal coordinates.
    y_exact : np.ndarray
        Analytical solution values.
    y_pred : np.ndarray
        Neural network predicted values.
    title : str
        Figure title.
    output_filename : Optional[str], default=None
        Path to save figure. If None, figure is not saved to disk.
    xlabel : str, default='x'
        Label for horizontal axis.
    ylabel : str, default=r'$\\Psi(x)$'
        Label for primary vertical axis.
    pred_label : str, default='Neural Network Solution'
        Legend label for predicted curve.
    exact_label : str, default='Analytical Solution'
        Legend label for exact curve.
    show : bool, default=True
        Whether to display the plot interactively.
    """
    setup_figure_style()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Top panel: Solution comparison
    ax1.plot(
        x, y_exact, label=exact_label, color="royalblue", linewidth=3.5, alpha=0.6
    )
    ax1.plot(
        x,
        y_pred,
        linestyle="--",
        color="crimson",
        linewidth=1.8,
        marker="o",
        markersize=6,
        markerfacecolor="none",
        markeredgecolor="crimson",
        label=pred_label,
    )
    ax1.set_title(title, fontweight="bold", pad=12)
    ax1.set_ylabel(ylabel)
    ax1.legend(loc="best", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Bottom panel: Pointwise absolute error
    abs_error = np.abs(y_pred - y_exact)
    ax2.plot(x, abs_error, color="darkorange", linewidth=1.8, label="Absolute Error")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("|Error|")
    ax2.set_yscale("log")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="best", framealpha=0.9)

    plt.tight_layout()

    if output_filename:
        ensure_dir(os.path.dirname(os.path.abspath(output_filename)))
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {output_filename}")

    if show:
        plt.show()
    else:
        plt.close(fig)
