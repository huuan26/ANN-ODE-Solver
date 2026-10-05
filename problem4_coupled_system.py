"""
Lagaris Problem 4: Coupled System of Two Non-Linear First-Order ODEs.

Reference:
    I. E. Lagaris, A. Likas, and D. I. Fotiadis,
    "Artificial Neural Networks for Solving Ordinary and Partial Differential Equations,"
    IEEE Transactions on Neural Networks, Vol. 9, No. 5, September 1998, pp. 987–1000.

Mathematical Formulation:
    Coupled non-linear system:
        dPsi_1/dx = cos(x) + Psi_1^2 + Psi_2 - (1 + x^2 + sin^2(x))
        dPsi_2/dx = 2*x - (1 + x^2)*sin(x) + Psi_1 * Psi_2
    Domain:
        x in [0, 3.0]
    Initial conditions:
        Psi_1(0) = 0.0
        Psi_2(0) = 1.0

Trial Solutions Formulation:
    Psi_{1,t}(x) = x * N_1(x, p)
    Psi_{2,t}(x) = 1.0 + x * N_2(x, p)
    where N(x, p) = [N_1(x), N_2(x)]^T is a 2-output deep neural network.
    Both initial conditions are analytically satisfied for any network weights.

Analytical Ground Truth:
    Psi_{1,exact}(x) = sin(x)
    Psi_{2,exact}(x) = 1.0 + x^2
"""

import argparse
import os
import sys
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

try:
    from src.models import MultiOutputDNN
    from src.utils import compute_metrics, ensure_dir, set_seed
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.models import MultiOutputDNN
    from src.utils import compute_metrics, ensure_dir, set_seed


def trial_solution(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trial functions for the 2-variable ODE system.

    Psi_{1,t}(x) = x * N_1(x)
    Psi_{2,t}(x) = 1 + x * N_2(x)
    """
    out = model(x)
    n1 = out[:, 0:1]
    n2 = out[:, 1:2]

    psi1_t = x * n1
    psi2_t = 1.0 + x * n2
    return psi1_t, psi2_t


def compute_loss(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Compute joint residual loss for both differential equations in the system.
    """
    x.requires_grad_(True)
    psi1_t, psi2_t = trial_solution(model, x)

    # Autograd derivatives
    d_psi1_t = torch.autograd.grad(
        outputs=psi1_t,
        inputs=x,
        grad_outputs=torch.ones_like(psi1_t),
        create_graph=True,
    )[0]

    d_psi2_t = torch.autograd.grad(
        outputs=psi2_t,
        inputs=x,
        grad_outputs=torch.ones_like(psi2_t),
        create_graph=True,
    )[0]

    res_1 = (
        d_psi1_t
        - torch.cos(x)
        - psi1_t**2
        - psi2_t
        + 1.0
        + x**2
        + torch.sin(x)**2
    )
    res_2 = (
        d_psi2_t
        - 2.0 * x
        + (1.0 + x**2) * torch.sin(x)
        - psi1_t * psi2_t
    )

    loss = torch.mean(res_1**2) + torch.mean(res_2**2)
    return loss


def exact_solution(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Analytical solutions: Psi_1(x) = sin(x), Psi_2(x) = 1 + x^2."""
    psi1_exact = np.sin(x)
    psi2_exact = 1.0 + x**2
    return psi1_exact, psi2_exact


def train_pinn_system(
    num_points: int = 200,
    hidden_dims: List[int] = [64, 64, 64],
    epochs_phase1: int = 20000,
    epochs_phase2: int = 20000,
    lr: float = 0.005,
    log_interval: int = 5000,
) -> nn.Module:
    """
    Two-stage curriculum training for the coupled non-linear system.

    Phase 1: Initial training on sub-domain [0, 1.5]
    Phase 2: Extension and convergence on full domain [0, 3.0]
    """
    model = MultiOutputDNN(hidden_dims=hidden_dims, input_dim=1, num_outputs=2)
    total_epochs = epochs_phase1 + epochs_phase2

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-5
    )

    print(f"Starting ANN curriculum training for Coupled ODE System...")
    print(f"Architecture: 1 -> {' -> '.join(map(str, hidden_dims))} -> 2 (Tanh)")
    print(f"Phase 1: Domain [0, 1.5] ({epochs_phase1} epochs)")
    print(f"Phase 2: Domain [0, 3.0] ({epochs_phase2} epochs)")
    print("-" * 60)

    # Phase 1: Sub-domain [0, 1.5]
    x_train_p1 = torch.linspace(0, 1.5, num_points, dtype=torch.float32).reshape(-1, 1)
    for i in range(epochs_phase1):
        optimizer.zero_grad()
        loss = compute_loss(model, x_train_p1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if i == 0 or (i + 1) % log_interval == 0:
            print(f"Phase 1 | Iter {i+1:5d}/{epochs_phase1} | Loss: {loss.item():.4e}")

    # Phase 2: Full domain [0, 3.0]
    x_train_p2 = torch.linspace(0, 3.0, num_points, dtype=torch.float32).reshape(-1, 1)
    for i in range(epochs_phase2):
        optimizer.zero_grad()
        loss = compute_loss(model, x_train_p2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if i == 0 or (i + 1) % log_interval == 0:
            print(f"Phase 2 | Iter {i+1:5d}/{epochs_phase2} | Loss: {loss.item():.4e}")

    print("-" * 60)
    print(f"Training completed. Final residual cost: {loss.item():.4e}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Solve Lagaris Problem 4 (Coupled Non-Linear System) using ANN."
    )
    parser.add_argument("--p1-epochs", type=int, default=20000, help="Phase 1 epochs")
    parser.add_argument("--p2-epochs", type=int, default=20000, help="Phase 2 epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--points", type=int, default=200, help="Evaluation points")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="figures/problem4_system_comparison.png",
        help="Path to save output figure",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display plot interactively")
    args = parser.parse_args()

    set_seed(args.seed)

    hidden_dims = [64, 64, 64]
    model = train_pinn_system(
        num_points=args.points,
        hidden_dims=hidden_dims,
        epochs_phase1=args.p1_epochs,
        epochs_phase2=args.p2_epochs,
        lr=args.lr,
    )

    x_np = np.linspace(0, 3.0, args.points).reshape(-1, 1)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)

    with torch.no_grad():
        psi1_pred, psi2_pred = trial_solution(model, x_tensor)
        psi1_pred = psi1_pred.numpy()
        psi2_pred = psi2_pred.numpy()

    psi1_exact, psi2_exact = exact_solution(x_np)

    metrics_1 = compute_metrics(psi1_pred, psi1_exact)
    metrics_2 = compute_metrics(psi2_pred, psi2_exact)

    print("\nQuantitative Evaluation Metrics:")
    print("  State Variable Psi_1(x) = sin(x):")
    print(f"    Max Absolute Error: {metrics_1['max_abs_error']:.4e}")
    print(f"    RMSE:               {metrics_1['root_mean_squared_error']:.4e}")
    print(f"    Relative L2 Error:  {metrics_1['relative_l2_error']:.4e}")
    print("  State Variable Psi_2(x) = 1 + x^2:")
    print(f"    Max Absolute Error: {metrics_2['max_abs_error']:.4e}")
    print(f"    RMSE:               {metrics_2['root_mean_squared_error']:.4e}")
    print(f"    Relative L2 Error:  {metrics_2['relative_l2_error']:.4e}")

    # Generate publication-grade 2-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Psi_1
    ax1.plot(
        x_np,
        psi1_exact,
        label=r"Analytical $\Psi_{1,exact}(x) = \sin(x)$",
        color="royalblue",
        linewidth=4.0,
        alpha=0.45,
    )
    ax1.plot(
        x_np,
        psi1_pred,
        linestyle="--",
        color="crimson",
        linewidth=1.8,
        marker="o",
        markersize=5,
        markerfacecolor="none",
        markeredgecolor="crimson",
        label=r"Neural Network $\Psi_{1,t}(x)$",
    )
    ax1.set_title(r"State Variable $\Psi_1(x)$", fontsize=12, fontweight="bold")
    ax1.set_xlabel("x", fontsize=11)
    ax1.set_ylabel(r"$\Psi_1(x)$", fontsize=11)
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Right: Psi_2
    ax2.plot(
        x_np,
        psi2_exact,
        label=r"Analytical $\Psi_{2,exact}(x) = 1 + x^2$",
        color="forestgreen",
        linewidth=4.0,
        alpha=0.45,
    )
    ax2.plot(
        x_np,
        psi2_pred,
        linestyle="--",
        color="darkorange",
        linewidth=1.8,
        marker="s",
        markersize=5,
        markerfacecolor="none",
        markeredgecolor="darkorange",
        label=r"Neural Network $\Psi_{2,t}(x)$",
    )
    ax2.set_title(r"State Variable $\Psi_2(x)$", fontsize=12, fontweight="bold")
    ax2.set_xlabel("x", fontsize=11)
    ax2.set_ylabel(r"$\Psi_2(x)$", fontsize=11)
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        "Lagaris Problem 4: Coupled Non-Linear ODE System Solutions",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    if args.output:
        ensure_dir(os.path.dirname(os.path.abspath(args.output)))
        plt.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to: {args.output}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
