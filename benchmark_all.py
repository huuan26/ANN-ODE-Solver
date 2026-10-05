"""
Benchmark Suite: Run all ODE problems and generate consolidated metrics and figures.

Usage:
    python benchmark_all.py          # Full precision runs
    python benchmark_all.py --quick  # Fast validation runs (fewer iterations)
"""

import argparse
import os
import sys
import time
from typing import Dict, List
import numpy as np
import torch

from src.utils import ensure_dir, set_seed
import problem1_first_order
import problem2_oscillatory
import problem3_second_order
import problem4_coupled_system
import euler_vs_pinn
import ode_decay_single_layer
import ode_decay_deep


def run_benchmarks(quick_mode: bool = False) -> List[Dict[str, str]]:
    ensure_dir("figures")
    set_seed(42)

    results = []
    print("=" * 75)
    print("ANN-ODE-Solver: Comprehensive Benchmark Execution")
    print(f"Mode: {'Quick Validation' if quick_mode else 'Full Training Run'}")
    print("=" * 75)

    # 1. Problem 1
    print("\n[1/6] Running Lagaris Problem 1 (First-Order Rational ODE)...")
    iters_p1 = 5000 if quick_mode else 40000
    t0 = time.time()
    x1_np = np.linspace(0, 2, 20).reshape(-1, 1)
    x1_tensor = torch.tensor(x1_np, dtype=torch.float32)
    m1 = problem1_first_order.train_pinn(
        x1_tensor, hidden_dims=[20, 20], num_iterations=iters_p1, learning_rate=0.001
    )
    with torch.no_grad():
        y1_pred = problem1_first_order.trial_solution(m1, x1_tensor, psi0=1.0).numpy()
    y1_exact = problem1_first_order.exact_solution(x1_np)
    met1 = problem1_first_order.compute_metrics(y1_pred, y1_exact)
    dt1 = time.time() - t0
    results.append({
        "Benchmark": "Problem 1 (Rational 1st-Order)",
        "Domain": "[0, 2]",
        "Max Abs Error": f"{met1['max_abs_error']:.2e}",
        "RMSE": f"{met1['root_mean_squared_error']:.2e}",
        "Rel L2 Error": f"{met1['relative_l2_error']:.2e}",
        "Time (s)": f"{dt1:.1f}",
    })

    # 2. Problem 2
    print("\n[2/6] Running Lagaris Problem 2 (Oscillatory ODE)...")
    iters_p2 = 5000 if quick_mode else 40000
    t0 = time.time()
    x2_np = np.linspace(0, 10, 50).reshape(-1, 1)
    x2_tensor = torch.tensor(x2_np, dtype=torch.float32)
    m2 = problem2_oscillatory.train_pinn(
        x2_tensor, hidden_dims=[20, 20], num_iterations=iters_p2, learning_rate=0.001
    )
    with torch.no_grad():
        y2_pred = problem2_oscillatory.trial_solution(m2, x2_tensor).numpy()
    y2_exact = problem2_oscillatory.exact_solution(x2_np)
    met2 = problem2_oscillatory.compute_metrics(y2_pred, y2_exact)
    dt2 = time.time() - t0
    results.append({
        "Benchmark": "Problem 2 (Oscillatory 1st-Order)",
        "Domain": "[0, 10]",
        "Max Abs Error": f"{met2['max_abs_error']:.2e}",
        "RMSE": f"{met2['root_mean_squared_error']:.2e}",
        "Rel L2 Error": f"{met2['relative_l2_error']:.2e}",
        "Time (s)": f"{dt2:.1f}",
    })

    # 3. Problem 3
    print("\n[3/6] Running Lagaris Problem 3 (Second-Order Damped ODE)...")
    iters_p3 = 5000 if quick_mode else 40000
    t0 = time.time()
    x3_np = np.linspace(0, 10, 100).reshape(-1, 1)
    x3_tensor = torch.tensor(x3_np, dtype=torch.float32)
    m3 = problem3_second_order.train_pinn(
        x3_tensor, hidden_dims=[32, 32, 32], num_iterations=iters_p3, learning_rate=0.002
    )
    with torch.no_grad():
        y3_pred = problem3_second_order.trial_solution(m3, x3_tensor).numpy()
    y3_exact = problem3_second_order.exact_solution(x3_np)
    met3 = problem3_second_order.compute_metrics(y3_pred, y3_exact)
    dt3 = time.time() - t0
    results.append({
        "Benchmark": "Problem 3 (Damped 2nd-Order)",
        "Domain": "[0, 10]",
        "Max Abs Error": f"{met3['max_abs_error']:.2e}",
        "RMSE": f"{met3['root_mean_squared_error']:.2e}",
        "Rel L2 Error": f"{met3['relative_l2_error']:.2e}",
        "Time (s)": f"{dt3:.1f}",
    })

    # 4. Problem 4
    print("\n[4/6] Running Lagaris Problem 4 (Coupled Non-Linear System)...")
    p1_ep = 2500 if quick_mode else 20000
    p2_ep = 2500 if quick_mode else 20000
    t0 = time.time()
    m4 = problem4_coupled_system.train_pinn_system(
        num_points=200,
        hidden_dims=[64, 64, 64],
        epochs_phase1=p1_ep,
        epochs_phase2=p2_ep,
        lr=0.005,
    )
    x4_np = np.linspace(0, 3, 200).reshape(-1, 1)
    x4_tensor = torch.tensor(x4_np, dtype=torch.float32)
    with torch.no_grad():
        psi1_pred, psi2_pred = problem4_coupled_system.trial_solution(m4, x4_tensor)
        psi1_pred = psi1_pred.numpy()
        psi2_pred = psi2_pred.numpy()
    psi1_exact, psi2_exact = problem4_coupled_system.exact_solution(x4_np)
    met4_1 = problem4_coupled_system.compute_metrics(psi1_pred, psi1_exact)
    met4_2 = problem4_coupled_system.compute_metrics(psi2_pred, psi2_exact)
    max_sys_err = max(met4_1["max_abs_error"], met4_2["max_abs_error"])
    dt4 = time.time() - t0
    results.append({
        "Benchmark": "Problem 4 (Coupled System: Psi1, Psi2)",
        "Domain": "[0, 3]",
        "Max Abs Error": f"{max_sys_err:.2e}",
        "RMSE": f"{max(met4_1['root_mean_squared_error'], met4_2['root_mean_squared_error']):.2e}",
        "Rel L2 Error": f"{max(met4_1['relative_l2_error'], met4_2['relative_l2_error']):.2e}",
        "Time (s)": f"{dt4:.1f}",
    })

    # 5. Euler vs ANN
    print("\n[5/6] Running Euler vs ANN Comparison...")
    t0 = time.time()
    xe_np = np.linspace(0, 1, 20).reshape(-1, 1)
    xe_tensor = torch.tensor(xe_np, dtype=torch.float32)
    me = euler_vs_pinn.train_pinn(
        epochs_p1=1500 if quick_mode else 5000,
        epochs_p2=1500 if quick_mode else 5000,
    )
    with torch.no_grad():
        ye_pred = euler_vs_pinn.trial_solution(me, xe_tensor).numpy()
    ye_exact = np.exp(-2.0 * xe_np)
    mete = euler_vs_pinn.compute_metrics(ye_pred, ye_exact)
    dt_e = time.time() - t0
    results.append({
        "Benchmark": "Exponential Decay (ANN)",
        "Domain": "[0, 1]",
        "Max Abs Error": f"{mete['max_abs_error']:.2e}",
        "RMSE": f"{mete['root_mean_squared_error']:.2e}",
        "Rel L2 Error": f"{mete['relative_l2_error']:.2e}",
        "Time (s)": f"{dt_e:.1f}",
    })

    # 6. Deep Decay
    print("\n[6/6] Running Deep Decay Solver...")
    t0 = time.time()
    xd_np = np.linspace(0, 1, 10).reshape(-1, 1)
    xd_tensor = torch.tensor(xd_np, dtype=torch.float32)
    md = ode_decay_deep.train_deep_pinn(
        xd_tensor, hidden_dims=[10, 10], num_iterations=2500 if quick_mode else 10000
    )
    with torch.no_grad():
        yd_pred = ode_decay_deep.trial_solution(md, xd_tensor).numpy()
    yd_exact = ode_decay_deep.exact_solution(xd_np)
    metd = ode_decay_deep.compute_metrics(yd_pred, yd_exact)
    dt_d = time.time() - t0
    results.append({
        "Benchmark": "Exponential Decay (Deep MLP)",
        "Domain": "[0, 1]",
        "Max Abs Error": f"{metd['max_abs_error']:.2e}",
        "RMSE": f"{metd['root_mean_squared_error']:.2e}",
        "Rel L2 Error": f"{metd['relative_l2_error']:.2e}",
        "Time (s)": f"{dt_d:.1f}",
    })

    # Summary Table
    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY RESULTS TABLE")
    print("=" * 90)
    header = f"{'Benchmark Problem':<38} | {'Domain':<8} | {'Max Abs Err':<12} | {'RMSE':<10} | {'Rel L2':<10} | {'Time (s)':<8}"
    print(header)
    print("-" * 90)
    for r in results:
        line = f"{r['Benchmark']:<38} | {r['Domain']:<8} | {r['Max Abs Error']:<12} | {r['RMSE']:<10} | {r['Rel L2 Error']:<10} | {r['Time (s)']:<8}"
        print(line)
    print("=" * 90)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run complete ANN ODE benchmark suite.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation benchmark with reduced iterations",
    )
    args = parser.parse_args()
    run_benchmarks(quick_mode=args.quick)


if __name__ == "__main__":
    main()
