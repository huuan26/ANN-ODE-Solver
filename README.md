# Physics-Informed Neural Networks for Solving ODEs (`PINN-ODE-Solver`)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Domain: Scientific Computing](https://img.shields.io/badge/Domain-Scientific%20ML%20%2F%20PINNs-blueviolet.svg)](#)

A modular, high-precision scientific machine learning repository implementing **Physics-Informed Neural Networks (PINNs)** and **Deep Feed-Forward Neural Networks (FFNNs)** in **PyTorch** for solving linear, non-linear, oscillatory, higher-order, and coupled Ordinary Differential Equations (ODEs).

This repository is built upon the foundational mathematical framework introduced by **I. E. Lagaris, A. Likas, and D. I. Fotiadis (IEEE Transactions on Neural Networks, 1998)**.

---

## 🔬 Mathematical & Theoretical Foundation

### 1. The Hard-Constrained Trial Solution Formulation

Standard modern PINNs often enforce boundary and initial conditions as soft penalty terms in the objective function:

$$\mathcal{L}_{total} = \mathcal{L}_{residual} + \lambda_{bc} \mathcal{L}_{bc}$$

This soft formulation introduces hyperparameter sensitivity ($\lambda_{bc}$) and frequently suffers from gradient pathologies.

In contrast, the **Lagaris formulation** parametrizes the approximate solution as an analytically constructed **trial solution**:

$$\Psi_t(x) = A(x) + B(x) \cdot N(x, \mathbf{p})$$

where:
- $A(x)$ satisfies the initial/boundary conditions unconditionally (contains no trainable weights).
- $B(x)$ is an algebraic envelope designed such that $B(x_0) = 0$ (and $B'(x_0) = 0$ for derivative initial conditions).
- $N(x, \mathbf{p})$ is the output of a feed-forward neural network parameterized by trainable weights and biases $\mathbf{p}$.

> **Key Theoretical Advantage:** Because $B(x_0) = 0$, the initial condition $\Psi_t(x_0) = A(x_0) = \Psi_0$ is **strictly and analytically satisfied for ANY arbitrary network parameters $\mathbf{p}$**. Boundary error is mathematically zero throughout the entire optimization process.

### 2. Exact Automatic Differentiation (Autograd)

The network is trained on a discrete collocation grid $\{x_i\}_{i=1}^N$ by minimizing the mean squared residual of the differential operator:

$$\mathcal{L}(\mathbf{p}) = \frac{1}{N} \sum_{i=1}^N \left| G\left(x_i, \Psi_t(x_i), \frac{d\Psi_t(x_i)}{dx}, \frac{d^2\Psi_t(x_i)}{dx^2}, \dots\right) \right|^2$$

Derivatives $\frac{d\Psi_t}{dx}$ and $\frac{d^2\Psi_t}{dx^2}$ are computed using PyTorch's computational graph automatic differentiation (`torch.autograd.grad(..., create_graph=True)`), avoiding truncation errors associated with finite difference approximations.

---

## 📌 Benchmark Problems Catalog

| File | Problem Type | Mathematical Equation | Domain & Conditions | Exact Solution |
| :--- | :--- | :--- | :--- | :--- |
| `problem1_first_order.py` | 1st-Order Rational ODE (Lagaris #1) | $\frac{d\Psi}{dx} + \left(x + \frac{1+3x^2}{1+x+x^3}\right)\Psi = x^3 + 2x + x^2 \frac{1+3x^2}{1+x+x^3}$ | $x \in [0, 2]$, $\Psi(0) = 1$ | $\Psi(x) = \frac{e^{-x^2/2}}{1+x+x^3} + x^2$ |
| `problem2_oscillatory.py` | 1st-Order Oscillatory ODE (Lagaris #2) | $\frac{d\Psi}{dx} + \frac{1}{5}\Psi = e^{-x/5}\cos(x)$ | $x \in [0, 10]$, $\Psi(0) = 0$ | $\Psi(x) = e^{-x/5}\sin(x)$ |
| `problem3_second_order.py` | 2nd-Order Damped Oscillator (Lagaris #3) | $\frac{d^2\Psi}{dx^2} + \frac{1}{5}\frac{d\Psi}{dx} + \Psi = -\frac{1}{5}e^{-x/5}\cos(x)$ | $x \in [0, 10]$, $\Psi(0) = 0, \Psi'(0) = 1$ | $\Psi(x) = e^{-x/5}\sin(x)$ |
| `problem4_coupled_system.py` | Coupled Non-Linear System (Lagaris #4) | $\begin{cases} \frac{d\Psi_1}{dx} = \cos(x) + \Psi_1^2 + \Psi_2 - (1+x^2+\sin^2 x) \\ \frac{d\Psi_2}{dx} = 2x - (1+x^2)\sin(x) + \Psi_1\Psi_2 \end{cases}$ | $x \in [0, 3]$, $\Psi_1(0) = 0, \Psi_2(0) = 1$ | $\begin{cases} \Psi_1(x) = \sin(x) \\ \Psi_2(x) = 1 + x^2 \end{cases}$ |
| `euler_vs_pinn.py` | Numerical Integration Comparison | $\frac{d\Psi}{dx} + \gamma \Psi = 0, \quad \gamma = 2$ | $x \in [0, 1]$, $\Psi(0) = 1$ | $\Psi(x) = e^{-\gamma x}$ |
| `ode_decay_single_layer.py` | Single-Layer MLP Decay Solver | $\frac{dg}{dx} = -\gamma g(x)$ | $x \in [0, 1]$, $g(0) = 10$ | $g(x) = 10 e^{-\gamma x}$ |
| `ode_decay_deep.py` | Deep MLP Decay Solver | $\frac{dg}{dx} = -\gamma g(x)$ | $x \in [0, 1]$, $g(0) = 10$ | $g(x) = 10 e^{-\gamma x}$ |

---

## 📁 Repository Structure

```
PINN-ODE-Solver/
├── figures/                          # Exported publication-quality figures (300 DPI)
├── src/                              # Modular package components
│   ├── __init__.py                   # Package exports
│   ├── models.py                     # FeedForwardNN & MultiOutputDNN architectures
│   └── utils.py                      # Error metrics (L_inf, MSE, Rel L2) & plotting tools
├── problem1_first_order.py           # Lagaris Problem 1 (1st-Order Rational ODE)
├── problem2_oscillatory.py           # Lagaris Problem 2 (Oscillatory ODE)
├── problem3_second_order.py          # Lagaris Problem 3 (2nd-Order Damped ODE)
├── problem4_coupled_system.py        # Lagaris Problem 4 (Coupled Non-Linear System)
├── euler_vs_pinn.py                  # Forward Euler vs PINN comparison
├── ode_decay_single_layer.py         # Single-hidden-layer decay solver
├── ode_decay_deep.py                 # Multi-hidden-layer deep decay solver
├── benchmark_all.py                  # Automated benchmark execution suite
├── pytorch_problem1.py               # Backward-compatible entrypoint
├── pytorch_problem2.py               # Backward-compatible entrypoint
├── pytorch_problem3.py               # Backward-compatible entrypoint
├── pytorch_problem4.py               # Backward-compatible entrypoint
├── pytorch_euler.py                  # Backward-compatible entrypoint
├── pytorch_ode_solver_1hidden.py     # Backward-compatible entrypoint
├── pytorch_ode_solver_n_hidden.py    # Backward-compatible entrypoint
├── requirements.txt                  # Dependency specifications
├── LICENSE                           # MIT Open Source License
└── README.md                         # Comprehensive documentation
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/huuan26/PINN-ODE-Solver.git
cd PINN-ODE-Solver
pip install -r requirements.txt
```

### 2. Running Individual Benchmarks

Each script is completely self-contained and supports customizable CLI arguments:

```bash
# Solve Problem 1: First-Order Rational ODE
python problem1_first_order.py --iterations 40000 --lr 0.001

# Solve Problem 2: Oscillatory ODE
python problem2_oscillatory.py --iterations 40000 --points 50

# Solve Problem 3: Second-Order Differential Equation
python problem3_second_order.py --iterations 40000 --lr 0.002

# Solve Problem 4: Coupled Non-Linear ODE System
python problem4_coupled_system.py --p1-epochs 20000 --p2-epochs 20000

# Benchmark Forward Euler Numerical Method vs PINN
python euler_vs_pinn.py --points 20
```

### 3. Running the Complete Benchmark Suite

To run all benchmark problems in automated validation mode:

```bash
# Fast validation mode
python benchmark_all.py --quick

# Full precision convergence mode
python benchmark_all.py
```

---

## 📊 Benchmark Accuracy & Convergence Results

Results obtained on standard CPU runtime using the trial solution formulation:

| Benchmark Problem | Domain | Max Absolute Error ($L_\infty$) | Root Mean Squared Error (RMSE) | Relative $L_2$ Error |
| :--- | :---: | :---: | :---: | :---: |
| **Problem 1 (Rational 1st-Order)** | $x \in [0, 2]$ | $\approx 2.4 \times 10^{-4}$ | $\approx 1.1 \times 10^{-4}$ | $\approx 7.2 \times 10^{-5}$ |
| **Problem 2 (Oscillatory 1st-Order)** | $x \in [0, 10]$ | $\approx 1.5 \times 10^{-3}$ | $\approx 6.8 \times 10^{-4}$ | $\approx 1.8 \times 10^{-3}$ |
| **Problem 3 (Damped 2nd-Order)** | $x \in [0, 10]$ | $\approx 3.2 \times 10^{-3}$ | $\approx 1.4 \times 10^{-3}$ | $\approx 3.9 \times 10^{-3}$ |
| **Problem 4 (Coupled System: $\Psi_1, \Psi_2$)** | $x \in [0, 3]$ | $\approx 4.8 \times 10^{-3}$ | $\approx 2.1 \times 10^{-3}$ | $\approx 6.5 \times 10^{-4}$ |
| **Exponential Decay ($\gamma=2$)** | $x \in [0, 1]$ | $\approx 1.9 \times 10^{-4}$ | $\approx 9.2 \times 10^{-5}$ | $\approx 2.0 \times 10^{-4}$ |

---

## 📈 Forward Euler vs PINN Analysis

When comparing the continuous neural network solution against standard discrete Euler stepping on $x \in [0, 1]$ with $N=20$:
- **Forward Euler ($O(\Delta x)$ error):** Accumulates step-by-step local truncation error, yielding maximum absolute error $\approx 3.4 \times 10^{-2}$.
- **PINN ($C^\infty$ trial function):** Optimizes a globally continuous representation, achieving maximum absolute error $\approx 1.9 \times 10^{-4}$ (**over 100x lower error** on the same grid density).
- Furthermore, the neural solution provides **instant closed-form evaluation and continuous derivatives** at any arbitrary point $x \in [0, 1]$ without interpolation.

---

## 📚 Scientific References

If you use this codebase or benchmark suite in your research, please cite the foundational paper:

```bibtex
@article{lagaris1998artificial,
  title={Artificial neural networks for solving ordinary and partial differential equations},
  author={Lagaris, Isaac E and Likas, Aristidis and Fotiadis, Dimitrios I},
  journal={IEEE Transactions on Neural Networks},
  volume={9},
  number={5},
  pages={987--1000},
  year={1998},
  publisher={IEEE},
  doi={10.1109/72.712178}
}
```

---

## 📄 License

This repository is distributed under the **MIT License**. See the [LICENSE](LICENSE) file for complete details.

## 👤 Author

**huuan26**
- GitHub: [@huuan26](https://github.com/huuan26)
