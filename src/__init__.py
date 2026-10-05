"""
ANN-ODE-Solver: Artificial Neural Networks and Feed-Forward Neural Networks
for solving Ordinary Differential Equations in PyTorch.

Based on the foundational method by Lagaris, Likas, and Fotiadis (IEEE TNN 1998).
"""

from .models import FeedForwardNN, MultiOutputDNN
from .utils import (
    compute_metrics,
    setup_figure_style,
    ensure_dir,
    set_seed,
)

__all__ = [
    "FeedForwardNN",
    "MultiOutputDNN",
    "compute_metrics",
    "setup_figure_style",
    "ensure_dir",
    "set_seed",
]
