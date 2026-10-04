"""
Neural Network Architectures for Differential Equation Solvers.

Provides multilayer perceptrons (MLP / FFNN) specifically designed for
Physics-Informed Neural Networks (PINNs) and trial solution approximations.
"""

from typing import List, Union
import torch
import torch.nn as nn


class FeedForwardNN(nn.Module):
    """
    Deep Feed-Forward Neural Network (Multilayer Perceptron) with smooth activations.

    Designed for approximating trial functions in ODE boundary/initial value problems.
    Uses Xavier uniform initialization to ensure stable autograd gradients during
    higher-order derivative calculations.

    Parameters
    ----------
    hidden_dims : List[int]
        List containing the number of hidden units in each intermediate layer.
        Example: [20, 20] or [32, 32, 32].
    input_dim : int, default=1
        Input spatial/temporal dimension (typically 1 for single-variable ODEs).
    output_dim : int, default=1
        Output dimension (scalar function approximation).
    activation : str, default='tanh'
        Activation function ('tanh', 'sigmoid', 'silu', 'relu').
        'tanh' is recommended for smooth differential operators.
    """

    def __init__(
        self,
        hidden_dims: List[int],
        input_dim: int = 1,
        output_dim: int = 1,
        activation: str = "tanh",
    ):
        super(FeedForwardNN, self).__init__()

        act_layer = {
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "silu": nn.SiLU,
            "relu": nn.ReLU,
        }.get(activation.lower(), nn.Tanh)

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_layer())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # Initialize weights using Xavier uniform distribution
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Xavier uniform initialization to linear layers."""
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates of shape (N, input_dim).

        Returns
        -------
        torch.Tensor
            Network outputs of shape (N, output_dim).
        """
        return self.network(x)


class MultiOutputDNN(nn.Module):
    """
    Multilayer Neural Network with multiple output heads for coupled ODE systems.

    Simultaneously approximates multiple coupled state variables (e.g., Psi_1(x), Psi_2(x))
    sharing a common feature representation.

    Parameters
    ----------
    hidden_dims : List[int]
        Number of units in each hidden layer, e.g. [64, 64, 64].
    input_dim : int, default=1
        Input dimension (e.g. 1 for coordinate x).
    num_outputs : int, default=2
        Number of state variables in the system.
    activation : str, default='tanh'
        Activation function ('tanh', 'sigmoid', 'silu').
    """

    def __init__(
        self,
        hidden_dims: List[int],
        input_dim: int = 1,
        num_outputs: int = 2,
        activation: str = "tanh",
    ):
        super(MultiOutputDNN, self).__init__()

        act_layer = {
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "silu": nn.SiLU,
        }.get(activation.lower(), nn.Tanh)

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_layer())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_outputs))
        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Xavier uniform initialization."""
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward evaluation.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates of shape (N, 1).

        Returns
        -------
        torch.Tensor
            Output vector of shape (N, num_outputs).
        """
        return self.network(x)
