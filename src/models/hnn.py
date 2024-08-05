"""
This module implements a Hamiltonian Neural Network (HNN) using PyTorch.

Classes:
    HNN: Implements the main Hamiltonian Neural Network model.

Dependencies:
    torch
    torch.nn
    torch.func
    typing
    matplotlib.pyplot
"""

from typing import List, Union
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.func import jacrev
from .utils import MLP

class HNN(nn.Module):
    """
    Implements a Hamiltonian Neural Network (HNN) model.

    This model learns the Hamiltonian dynamics of a system using an MLP as NN.
    """

    def __init__(
        self,
        n_elements: int,
        hidden_dims: Union[int, List[int]] = 200,
        num_layers: int = 3,
        baseline: bool = False,
        nonlinearity: str = 'softplus'
    ):
        """
        Initialize the Hamiltonian Neural Network.

        Args:
            n_elements: Number of elements in the input state.
            hidden_dims: Dimension(s) of hidden layers. Can be an int for uniform hidden dimensions
                         or a list to specify dimensions for each hidden layer.
            num_layers: Total number of layers in the MLP including input and output layers.
            baseline: Whether to use the baseline model.
            nonlinearity: Name of the activation function to use.
        """
        super(HNN, self).__init__()
        self.baseline = baseline
        self.differentiable_model = MLP(n_elements, hidden_dims, num_layers, baseline, nonlinearity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the HNN.

        Args:
            x: Input tensor of shape (batch_size, 2*n_elements).

        Returns:
            Output tensor or list of tensors.
        """

        if self.baseline:
            return self.differentiable_model(x)
    
        grad_H = torch.vmap(jacrev(self.differentiable_model))(x).squeeze(1)

        theta_dot = grad_H[:, x.shape[1]//2:] 
        p_dot = -grad_H[:, :x.shape[1]//2]

        return torch.cat([theta_dot, p_dot], dim=1)

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create a simple pendulum system
    n_elements = 1  # One angle and one momentum
    hnn = HNN(n_elements, hidden_dims=[64, 64], num_layers=4, baseline=False)

    # Generate some dummy data
    batch_size = 32
    x = torch.rand(batch_size, 2 * n_elements, requires_grad=True)

    def test_hnn(model, x, mode):
        print(f"\nTesting HNN in {mode} mode:")
        model.train() if mode == "training" else model.eval()
        
        # Ensure we can compute gradients even in eval mode
        with torch.set_grad_enabled(True):
            y = model(x)
    
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {y.shape}")
            
            # Compute and print gradients
            loss = y.sum()
            loss.backward()
            print(f"Gradient of x: {x.grad is not None}")
            print(f"Gradient of model parameters: {any(p.grad is not None for p in model.parameters())}")

        # Reset gradients
        x.grad = None
        model.zero_grad()

    # Test in training mode
    test_hnn(hnn, x, "training")

    # Test in evaluation mode
    test_hnn(hnn, x, "evaluation")

    # Visualize a trajectory
    def visualize_trajectory(model, x0, steps=100, dt=0.1):
        model.eval()
        trajectory = [x0.view(-1)]
        for _ in range(steps):
            x = trajectory[-1].view(1, -1).requires_grad_(True)
            with torch.enable_grad():
                dx = model(x).view(-1)
            x_new = x.detach().view(-1) + dx.detach() * dt
            trajectory.append(x_new)
        
        trajectory = torch.stack(trajectory)
        plt.figure(figsize=(10, 5))
        plt.plot(trajectory[:, 0], trajectory[:, 1])
        plt.xlabel('Angle')
        plt.ylabel('Momentum')
        plt.title('Phase Space Trajectory')
        plt.grid(True)
        plt.show()

    # Visualize a trajectory starting from a random point
    x0 = torch.rand(2 * n_elements)
    visualize_trajectory(hnn, x0)

    print("\nHNN test completed successfully!")