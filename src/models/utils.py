"""
This module implements a Multi-Layer Perceptron (MLP) and related utility functions.

The module includes:
- A function to choose activation functions (choose_nonlinearity)
- An MLP class with configurable input and output dimensions, number of layers, and hidden dimensions

Classes:
    MLP: Implements a Multi-Layer Perceptron for HNN task with configurable activation.

Dependencies:
    torch
    torch.nn
    typing
"""

from typing import Callable, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F

def choose_nonlinearity(name: str) -> Callable:
    """
    Select the nonlinearity function based on the given name.

    Args:
        name: Name of the nonlinearity function.

    Returns:
        The selected nonlinearity function.

    Raises:
        ValueError: If the nonlinearity is not recognized.
    """
    nl_dict = {
        'tanh': torch.tanh,
        'sigmoid': torch.sigmoid,
        'softplus': F.softplus,
        'selu': F.selu,
        'elu': F.elu,
        'swish': lambda x: x * torch.sigmoid(x)
    }
    
    if name not in nl_dict:
        raise ValueError("nonlinearity not recognized")
    
    return nl_dict[name]

class MLP(nn.Module):
    """
    Implements a Multi-Layer Perceptron with configurable activation, dimensions, and number of layers.
    """

    def __init__(self, n_elements: int, hidden_dims: Union[int, List[int]] = 200, 
                 num_layers: int = 3, baseline: bool = False, nonlinearity: str = 'tanh'):
        """
        Initialize the Multi-Layer Perceptron.

        Args:
            n_elements: Number of elements, used to determine input and output dimensions.
            hidden_dims: Dimension(s) of hidden layers. Can be an int for uniform hidden dimensions
                         or a list to specify dimensions for each hidden layer.
            num_layers: Total number of layers including input and output layers.
            baseline: If True, output dimension is 2*n_elements, else 1 (needed in the HNN model).
            nonlinearity: Name of the activation function to use.
        """
        super(MLP, self).__init__()
        self.input_dim = 2 * n_elements
        self.output_dim = 2 * n_elements if baseline else 1
        
        # Configure hidden dimensions
        if isinstance(hidden_dims, int):
            self.hidden_dims = [hidden_dims] * (num_layers - 1)
        elif isinstance(hidden_dims, list):
            if len(hidden_dims) != num_layers - 1:
                raise ValueError("Length of hidden_dims list must be equal to num_layers - 2")
            self.hidden_dims = hidden_dims
        else:
            raise TypeError("hidden_dims must be an int or a list of ints")

        # Create layers
        layers = []
        in_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.output_dim, bias=False))
        
        self.layers = nn.ModuleList(layers)
        
        # Xavier initialization for weights
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
        
        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        for layer in self.layers[:-1]:
            x = self.nonlinearity(layer(x))
        return self.layers[-1](x)