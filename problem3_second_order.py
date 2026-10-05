"""
Lagaris Problem 3: Second-Order Damped Harmonic Oscillator ODE.

Reference:
    I. E. Lagaris, A. Likas, and D. I. Fotiadis,
    "Artificial Neural Networks for Solving Ordinary and Partial Differential Equations,"
    IEEE Transactions on Neural Networks, Vol. 9, No. 5, September 1998, pp. 987–1000.

Mathematical Formulation:
    Differential equation:
        d^2Psi/dx^2 + (1/5) * dPsi/dx + Psi(x) = -(1/5) * exp(-x/5) * cos(x),  x in [0, 10]
    Initial conditions:
        Psi(0) = 0.0
        Psi'(0) = 1.0

Trial Solution Formulation:
    Psi_t(x) = x + x^2 * N(x, p)
    Verification of boundary conditions:
        Psi_t(0) = 0.0 + 0.0 = 0.0
        dPsi_t/dx = 1 + 2*x*N(x) + x^2 * dN/dx  ==>  dPsi_t/dx(0) = 1.0
    Both conditions are analytically satisfied without penalty hyperparameter tuning.

Analytical Ground Truth:
    Psi_exact(x) = exp(-x/5) * sin(x)
"""

import argparse
import os
import sys
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

try:
    from src.models import FeedForwardNN
    from src.utils import compute_metrics, ensure_dir, set_seed
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.models import FeedForwardNN
    from src.utils import compute_metrics, ensure_dir, set_seed


def trial_solution(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Trial function for second-order ODE satisfying Psi(0) = 0 and Psi'(0) = 1.

    Psi_t(x) = x + x^2 * N(x)
    """
    return x + (x**2) * model(x)


def compute_loss(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Compute mean squared residual using first and second autograd derivatives.

    Residual: d^2Psi_t/dx^2 + (1/5)*dPsi_t/dx + Psi_t + (1/5)*exp(-x/5)*cos(x) = 0
    """
    x.requires_grad_(True)
    psi_t = trial_solution(model, x)

    # First derivative dPsi_t / dx
    d_psi_t = torch.autograd.grad(
        outputs=psi_t,
        inputs=x,
        grad_outputs=torch.ones_like(psi_t),
        create_graph=True,
    )[0]

    # Second derivative d^2Psi_t / dx^2
    d2_psi_t = torch.autograd.grad(
        outputs=d_psi_t,
        inputs=x,
        grad_outputs=torch.ones_like(d_psi_t),
        create_graph=True,
    )[0]

    term_1 = d2_psi_t
    term_2 = (1.0 / 5.0) * d_psi_t
    term_3 = psi_t
    term_4 = (1.0 / 5.0) * torch.exp(-x / 5.0) * torch.cos(x)

    residual = term_1 + term_2 + term_3 + term_4
    loss = torch.mean(residual**2)
    return loss


def exact_solution(x: np.ndarray) -> np.ndarray:
    """Analytical solution: Psi(x) = exp(-x/5) * sin(x)."""
    return np.exp(-x / 5.0) * np.sin(x)


def train_pinn(
    x_train: torch.Tensor,
    hidden_dims: List[int] = [32, 32, 32],
    num_iterations: int = 40000,
    learning_rate: float = 0.002,
    log_interval: int = 5000,
) -> nn.Module:
    """Train deep neural network for second-order ODE using Adam."""
    model = FeedForwardNN(hidden_dims=hidden_dims, activation="tanh")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting ANN training for 2nd-order ODE on {len(x_train)} collocation points...")
    print(f"Architecture: 1 -> {' -> '.join(map(str, hidden_dims))} -> 1 (Tanh)")
    print(f"Optimizer: Adam (lr={learning_rate})")
    print("-" * 55)

    for i in range(num_iterations):
        optimizer.zero_grad()
        loss = compute_loss(model, x_train)
        loss.backward()
        optimizer.step()

        if i == 0 or (i + 1) % log_interval == 0:
            print(f"Iter {i+1:5d}/{num_iterations} | Residual Loss: {loss.item():.4e}")

    print("-" * 55)
    print(f"Training completed. Final residual cost: {loss.item():.4e}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Solve Lagaris Problem 3 (Second-Order Damped ODE) using ANN."
    )
    parser.add_argument("--iterations", type=int, default=40000, help="Training iterations")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    parser.add_argument("--points", type=int, default=100, help="Collocation points")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="figures/problem3_comparison.png",
        help="Path to save output figure",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display plot interactively")
    args = parser.parse_args()

    set_seed(args.seed)

    x_np = np.linspace(0, 10, args.points).reshape(-1, 1)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)

    hidden_dims = [32, 32, 32]
    model = train_pinn(
        x_tensor,
        hidden_dims=hidden_dims,
        num_iterations=args.iterations,
        learning_rate=args.lr,
    )

    with torch.no_grad():
        y_pred = trial_solution(model, x_tensor).numpy()

    y_exact = exact_solution(x_np)

    metrics = compute_metrics(y_pred, y_exact)
    print("\nQuantitative Evaluation Metrics (ANN vs Exact):")
    print(f"  Max Absolute Error (L_inf): {metrics['max_abs_error']:.4e}")
    print(f"  Mean Squared Error (MSE):    {metrics['mean_squared_error']:.4e}")
    print(f"  Root Mean Squared Error:     {metrics['root_mean_squared_error']:.4e}")
    print(f"  Relative L2 Error:           {metrics['relative_l2_error']:.4e}")

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_np,
        y_exact,
        label=r"Analytical Solution $\Psi_{exact}(x) = e^{-x/5}\sin(x)$",
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
        markersize=6,
        markerfacecolor="none",
        markeredgecolor="red",
        label=r"Neural Network Trial Solution $\Psi_t(x) = x + x^2 \cdot N(x)$",
    )

    plt.title(
        "Lagaris Problem 3: Second-Order ODE Solution Comparison",
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
