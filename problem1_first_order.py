"""
Lagaris Problem 1: First-Order Non-Linear ODE with Rational Coefficients.

Reference:
    I. E. Lagaris, A. Likas, and D. I. Fotiadis,
    "Artificial Neural Networks for Solving Ordinary and Partial Differential Equations,"
    IEEE Transactions on Neural Networks, Vol. 9, No. 5, September 1998, pp. 987–1000.

Mathematical Formulation:
    Differential equation:
        dPsi/dx + P(x) * Psi(x) = Q(x),  x in [0, 2]
    where:
        P(x) = x + (1 + 3*x^2) / (1 + x + x^3)
        Q(x) = x^3 + 2*x + x^2 * (1 + 3*x^2) / (1 + x + x^3)
    Initial condition:
        Psi(0) = 1.0

Trial Solution Formulation:
    Psi_t(x) = 1.0 + x * N(x, p)
    Note: At x = 0, Psi_t(0) = 1.0 + 0 = 1.0 (analytically exact).

Analytical Ground Truth:
    Psi_exact(x) = exp(-x^2 / 2) / (1 + x + x^3) + x^2
"""

import argparse
import os
import sys
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Support both package and direct execution
try:
    from src.models import FeedForwardNN
    from src.utils import compute_metrics, ensure_dir, set_seed
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.models import FeedForwardNN
    from src.utils import compute_metrics, ensure_dir, set_seed


def trial_solution(model: nn.Module, x: torch.Tensor, psi0: float = 1.0) -> torch.Tensor:
    """
    Construct trial solution identically satisfying initial condition Psi(0) = psi0.

    Psi_t(x) = psi0 + x * N(x)
    """
    return psi0 + x * model(x)


def compute_loss(model: nn.Module, x: torch.Tensor, psi0: float = 1.0) -> torch.Tensor:
    """
    Compute mean squared residual of the differential equation.

    Residual: dPsi_t/dx + P(x)*Psi_t - Q(x) = 0
    """
    x.requires_grad_(True)
    psi_t = trial_solution(model, x, psi0=psi0)

    # First derivative using automatic differentiation
    d_psi_t = torch.autograd.grad(
        outputs=psi_t,
        inputs=x,
        grad_outputs=torch.ones_like(psi_t),
        create_graph=True,
    )[0]

    p_x = x + (1.0 + 3.0 * x**2) / (1.0 + x + x**3)
    q_x = x**3 + 2.0 * x + (x**2) * (1.0 + 3.0 * x**2) / (1.0 + x + x**3)

    residual = d_psi_t + p_x * psi_t - q_x
    loss = torch.mean(residual**2)
    return loss


def exact_solution(x: np.ndarray) -> np.ndarray:
    """Calculate the exact analytical solution."""
    term1 = np.exp(-(x**2) / 2.0) / (1.0 + x + x**3)
    term2 = x**2
    return term1 + term2


def train_pinn(
    x_train: torch.Tensor,
    hidden_dims: List[int] = [20, 20],
    num_iterations: int = 40000,
    learning_rate: float = 0.001,
    log_interval: int = 5000,
) -> nn.Module:
    """
    Train Feed-Forward Neural Network to solve the ODE.

    Parameters
    ----------
    x_train : torch.Tensor
        Collocation grid points.
    hidden_dims : List[int]
        Layer sizes for the MLP.
    num_iterations : int
        Number of optimization steps.
    learning_rate : float
        SGD learning rate.
    log_interval : int
        Frequency of logging iterations.

    Returns
    -------
    nn.Module
        Trained PyTorch model.
    """
    model = FeedForwardNN(hidden_dims=hidden_dims, activation="tanh")
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    print(f"Starting PINN training on {len(x_train)} collocation points...")
    print(f"Architecture: 1 -> {' -> '.join(map(str, hidden_dims))} -> 1 (Tanh)")
    print(f"Optimizer: Momentum SGD (lr={learning_rate}, momentum=0.9)")
    print("-" * 55)

    for i in range(num_iterations):
        optimizer.zero_grad()
        loss = compute_loss(model, x_train, psi0=1.0)
        loss.backward()

        # Gradient clipping to ensure optimization stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if i == 0 or (i + 1) % log_interval == 0:
            print(f"Iter {i+1:5d}/{num_iterations} | Residual Loss: {loss.item():.4e}")

    print("-" * 55)
    print(f"Training completed. Final residual cost: {loss.item():.4e}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Solve Lagaris Problem 1 using Physics-Informed Neural Network."
    )
    parser.add_argument("--iterations", type=int, default=40000, help="Training iterations")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--points", type=int, default=20, help="Collocation points")
    parser.add_argument("--seed", type=int, default=15, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="figures/problem1_comparison.png",
        help="Path to save output figure",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display plot interactively")
    args = parser.parse_args()

    set_seed(args.seed)

    # Collocation grid on [0, 2]
    x_np = np.linspace(0, 2, args.points).reshape(-1, 1)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)

    hidden_dims = [20, 20]
    model = train_pinn(
        x_tensor,
        hidden_dims=hidden_dims,
        num_iterations=args.iterations,
        learning_rate=args.lr,
    )

    # Evaluate neural network prediction
    with torch.no_grad():
        y_pred = trial_solution(model, x_tensor, psi0=1.0).numpy()

    y_exact = exact_solution(x_np)

    # Quantitative error metrics
    metrics = compute_metrics(y_pred, y_exact)
    print("\nQuantitative Evaluation Metrics (PINN vs Exact):")
    print(f"  Max Absolute Error (L_inf): {metrics['max_abs_error']:.4e}")
    print(f"  Mean Squared Error (MSE):    {metrics['mean_squared_error']:.4e}")
    print(f"  Root Mean Squared Error:     {metrics['root_mean_squared_error']:.4e}")
    print(f"  Relative L2 Error:           {metrics['relative_l2_error']:.4e}")

    # Generate publication-quality figure
    plt.figure(figsize=(10, 6))
    plt.plot(
        x_np,
        y_exact,
        label=r"Analytical Solution $\Psi_{exact}(x) = \frac{e^{-x^2/2}}{1+x+x^3} + x^2$",
        color="royalblue",
        linewidth=4.0,
        alpha=0.45,
    )
    plt.plot(
        x_np,
        y_pred,
        linestyle="--",
        color="red",
        linewidth=1.8,
        marker="o",
        markersize=7,
        markerfacecolor="none",
        markeredgecolor="red",
        label=r"Neural Network Trial Solution $\Psi_t(x) = 1 + x \cdot N(x)$",
    )

    plt.title(
        "Lagaris Problem 1: First-Order ODE Solution Comparison",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    plt.xlabel("Domain coordinate x", fontsize=11)
    plt.ylabel(r"Solution $\Psi(x)$", fontsize=11)
    plt.legend(fontsize=11, loc="best", framealpha=0.9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if args.output:
        ensure_dir(os.path.dirname(os.path.abspath(args.output)))
        plt.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to: {args.output}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
