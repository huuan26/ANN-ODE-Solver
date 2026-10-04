"""
Single Hidden Layer Neural Network ODE Solver for Exponential Decay.

Problem:
    First-order linear ODE:
        dg/dx = -gamma * g(x),  x in [0, 1]
    Initial condition:
        g(0) = g0
    Parameters:
        gamma = 2.0, g0 = 10.0

Trial Function:
    g_t(x) = g0 + x * N(x, p)

Architecture:
    Single hidden layer with Sigmoid activation functions:
    Linear(1 -> H) -> Sigmoid -> Linear(H -> 1)
"""

import argparse
import os
import sys
from typing import Optional
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


class SingleLayerNN(nn.Module):
    """Single hidden layer neural network with Sigmoid activation."""

    def __init__(self, hidden_dim: int = 10):
        super(SingleLayerNN, self).__init__()
        self.hidden = nn.Linear(1, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.sigmoid(self.hidden(x)))


def trial_solution(model: nn.Module, x: torch.Tensor, g0: float = 10.0) -> torch.Tensor:
    """Trial solution identically satisfying g(0) = g0: g_t(x) = g0 + x * N(x)."""
    return g0 + x * model(x)


def compute_loss(
    model: nn.Module, x: torch.Tensor, gamma: float = 2.0, g0: float = 10.0
) -> torch.Tensor:
    """Compute MSE loss of residual: dg_t/dx + gamma * g_t = 0."""
    x.requires_grad_(True)
    g_t = trial_solution(model, x, g0=g0)

    d_g_t = torch.autograd.grad(
        outputs=g_t,
        inputs=x,
        grad_outputs=torch.ones_like(g_t),
        create_graph=True,
    )[0]

    residual = d_g_t + gamma * g_t
    return torch.mean(residual**2)


def exact_solution(x: np.ndarray, gamma: float = 2.0, g0: float = 10.0) -> np.ndarray:
    """Exact analytical solution: g(x) = g0 * exp(-gamma * x)."""
    return g0 * np.exp(-gamma * x)


def train_single_layer(
    x_train: torch.Tensor,
    hidden_dim: int = 10,
    num_iterations: int = 10000,
    lr: float = 0.001,
    gamma: float = 2.0,
    g0: float = 10.0,
) -> nn.Module:
    """Train single-layer neural network using SGD."""
    model = SingleLayerNN(hidden_dim=hidden_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    print(f"Training Single-Layer NN (H={hidden_dim}) for exponential decay...")
    for i in range(num_iterations):
        optimizer.zero_grad()
        loss = compute_loss(model, x_train, gamma=gamma, g0=g0)
        loss.backward()
        optimizer.step()

        if i == 0 or (i + 1) % 2500 == 0:
            print(f"Iter {i+1:5d}/{num_iterations} | Residual Loss: {loss.item():.4e}")

    print(f"Training completed. Final cost: {loss.item():.4e}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Solve exponential decay ODE using a single-layer neural network."
    )
    parser.add_argument("--iterations", type=int, default=10000, help="Training iterations")
    parser.add_argument("--hidden", type=int, default=10, help="Number of hidden neurons")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--points", type=int, default=10, help="Collocation points")
    parser.add_argument("--gamma", type=float, default=2.0, help="Decay parameter gamma")
    parser.add_argument("--g0", type=float, default=10.0, help="Initial condition g(0)")
    parser.add_argument("--seed", type=int, default=15, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="figures/decay_single_layer_comparison.png",
        help="Path to save output figure",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display plot interactively")
    args = parser.parse_args()

    set_seed(args.seed)

    x_np = np.linspace(0, 1, args.points).reshape(-1, 1)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)

    model = train_single_layer(
        x_tensor,
        hidden_dim=args.hidden,
        num_iterations=args.iterations,
        lr=args.lr,
        gamma=args.gamma,
        g0=args.g0,
    )

    with torch.no_grad():
        res_nn = trial_solution(model, x_tensor, g0=args.g0).numpy()

    res_exact = exact_solution(x_np, gamma=args.gamma, g0=args.g0)
    metrics = compute_metrics(res_nn, res_exact)

    print("\nQuantitative Evaluation Metrics:")
    print(f"  Max Absolute Error: {metrics['max_abs_error']:.4e}")
    print(f"  MSE:                {metrics['mean_squared_error']:.4e}")
    print(f"  RMSE:               {metrics['root_mean_squared_error']:.4e}")
    print(f"  Relative L2 Error:  {metrics['relative_l2_error']:.4e}")

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_np,
        res_exact,
        label=r"Analytical Solution $g(x) = g_0 e^{-\gamma x}$",
        color="royalblue",
        linewidth=5.0,
        alpha=0.45,
    )
    plt.plot(
        x_np,
        res_nn,
        linestyle="--",
        color="crimson",
        linewidth=1.8,
        marker="o",
        markersize=7,
        markerfacecolor="none",
        markeredgecolor="crimson",
        label=r"Single-Layer NN Trial Solution",
    )

    plt.title(
        "Single-Layer NN Solving Exponential Decay ODE",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    plt.xlabel("x", fontsize=11)
    plt.ylabel("g(x)", fontsize=11)
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
