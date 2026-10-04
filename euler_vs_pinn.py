"""
Comparison: Classical Numerical Integration (Forward Euler) vs Neural Network (PINN).

Problem:
    Exponential Decay ODE:
        dPsi/dx + gamma * Psi(x) = 0,  x in [0, 1]
    Initial condition:
        Psi(0) = 1.0
    Parameters:
        gamma = 2.0

Methods Evaluated:
    1. Forward Euler Method:
       Psi_{i+1} = Psi_i - dx * gamma * Psi_i
       Order of accuracy: O(dx)
    2. Physics-Informed Neural Network:
       Trial solution: Psi_t(x) = 1.0 + x * N(x, p)
       Trained via continuous autograd residual minimization.
    3. Analytical Ground Truth:
       Psi(x) = exp(-gamma * x)
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


def trial_solution(model: nn.Module, x: torch.Tensor, psi0: float = 1.0) -> torch.Tensor:
    """Trial function satisfying Psi(0) = psi0: Psi_t(x) = psi0 + x * N(x)."""
    return psi0 + x * model(x)


def compute_loss(
    model: nn.Module, x: torch.Tensor, gamma: float = 2.0, psi0: float = 1.0
) -> torch.Tensor:
    """Compute ODE residual: dPsi_t/dx + gamma * Psi_t = 0."""
    x.requires_grad_(True)
    psi_t = trial_solution(model, x, psi0=psi0)

    d_psi_t = torch.autograd.grad(
        outputs=psi_t,
        inputs=x,
        grad_outputs=torch.ones_like(psi_t),
        create_graph=True,
    )[0]

    residual = d_psi_t + gamma * psi_t
    loss = torch.mean(residual**2)
    return loss


def solve_forward_euler(
    x: np.ndarray, gamma: float = 2.0, psi0: float = 1.0
) -> np.ndarray:
    """
    Standard Forward Euler numerical step integration.

    Psi_{i+1} = Psi_i - dx * gamma * Psi_i
    """
    n = len(x)
    dx = x[1] - x[0]
    psi_euler = np.zeros((n, 1))
    psi_euler[0] = psi0

    for i in range(1, n):
        psi_euler[i] = psi_euler[i - 1] - dx * gamma * psi_euler[i - 1]

    return psi_euler


def train_pinn(
    hidden_dims: List[int] = [32, 32],
    num_points: int = 20,
    t_max: float = 1.0,
    gamma: float = 2.0,
    psi0: float = 1.0,
    epochs_p1: int = 5000,
    epochs_p2: int = 5000,
    lr: float = 0.01,
) -> nn.Module:
    """Train neural network solver with two-phase domain refinement."""
    model = FeedForwardNN(hidden_dims=hidden_dims, activation="tanh")
    total_epochs = epochs_p1 + epochs_p2

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-5
    )

    # Phase 1: Train on sub-interval [0, t_max / 2]
    x_train_p1 = torch.linspace(0, t_max / 2.0, num_points, dtype=torch.float32).reshape(-1, 1)
    for _ in range(epochs_p1):
        optimizer.zero_grad()
        loss = compute_loss(model, x_train_p1, gamma=gamma, psi0=psi0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    # Phase 2: Train on full interval [0, t_max]
    x_train_p2 = torch.linspace(0, t_max, num_points, dtype=torch.float32).reshape(-1, 1)
    for _ in range(epochs_p2):
        optimizer.zero_grad()
        loss = compute_loss(model, x_train_p2, gamma=gamma, psi0=psi0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    final_loss = compute_loss(model, x_train_p2, gamma=gamma, psi0=psi0)
    print(f"PINN Training completed. Final residual cost: {final_loss.item():.4e}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Forward Euler vs Physics-Informed Neural Network."
    )
    parser.add_argument("--points", type=int, default=20, help="Collocation points")
    parser.add_argument("--gamma", type=float, default=2.0, help="Decay rate parameter")
    parser.add_argument("--psi0", type=float, default=1.0, help="Initial condition Psi(0)")
    parser.add_argument("--tmax", type=float, default=1.0, help="End time T")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="figures/euler_vs_pinn_comparison.png",
        help="Path to save output figure",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display plot interactively")
    args = parser.parse_args()

    set_seed(args.seed)

    x_np = np.linspace(0, args.tmax, args.points).reshape(-1, 1)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)

    # 1. Forward Euler
    psi_euler = solve_forward_euler(x_np, gamma=args.gamma, psi0=args.psi0)

    # 2. Physics-Informed Neural Network
    model = train_pinn(
        hidden_dims=[32, 32],
        num_points=args.points,
        t_max=args.tmax,
        gamma=args.gamma,
        psi0=args.psi0,
    )
    with torch.no_grad():
        psi_pinn = trial_solution(model, x_tensor, psi0=args.psi0).numpy()

    # 3. Analytical Ground Truth
    psi_exact = args.psi0 * np.exp(-args.gamma * x_np)

    metrics_euler = compute_metrics(psi_euler, psi_exact)
    metrics_pinn = compute_metrics(psi_pinn, psi_exact)

    print("\nBenchmark Comparison Results:")
    print("----------------------------------------------------------------")
    print(f"{'Metric':<25} | {'Forward Euler':<18} | {'PINN':<18}")
    print("----------------------------------------------------------------")
    print(
        f"{'Max Absolute Error':<25} | {metrics_euler['max_abs_error']:<18.4e} | {metrics_pinn['max_abs_error']:<18.4e}"
    )
    print(
        f"{'RMSE':<25} | {metrics_euler['root_mean_squared_error']:<18.4e} | {metrics_pinn['root_mean_squared_error']:<18.4e}"
    )
    print(
        f"{'Relative L2 Error':<25} | {metrics_euler['relative_l2_error']:<18.4e} | {metrics_pinn['relative_l2_error']:<18.4e}"
    )
    print("----------------------------------------------------------------")

    # High-resolution comparison plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        x_np,
        psi_exact,
        label=r"Analytical Solution $\Psi(x) = \Psi_0 e^{-\gamma x}$",
        color="royalblue",
        linewidth=4.0,
        alpha=0.45,
    )
    plt.plot(
        x_np,
        psi_euler,
        linestyle=":",
        color="forestgreen",
        linewidth=1.8,
        marker="^",
        markersize=7,
        markerfacecolor="none",
        markeredgecolor="forestgreen",
        label="Classical Forward Euler Integration",
    )
    plt.plot(
        x_np,
        psi_pinn,
        linestyle="--",
        color="crimson",
        linewidth=1.8,
        marker="o",
        markersize=7,
        markerfacecolor="none",
        markeredgecolor="crimson",
        label="Physics-Informed Neural Network",
    )

    plt.title(
        "Exponential Decay ODE: Forward Euler vs PINN vs Analytical Solution",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    plt.xlabel("Coordinate x", fontsize=11)
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
