"""
This module implements a dataset for simulating single pendulum trajectories using JAX, we add noisy to the original trajectory3.
It uses symplectic integration methods for numerical integration and supports vectorized operations for efficiency.
"""

import os
from typing import Tuple, Any, Optional
import jax
import jax.numpy as jnp
from jax import vmap, jit
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# Relative imports within the package, I choose this strategy
from src.single_pendulum.config import single_pendulum_config as config
from src.single_pendulum.utils.plotting import animate_single_pendulum_with_energy_and_phase
from src.common.utils.integrators import (
    symplectic_integrator_step,
    single_pendulum_position_derivative,
    single_pendulum_momentum_derivative,
)

from functools import partial

jax.config.update("jax_enable_x64", True) # During data simulation, we use 64-bit precision

class SinglePendulumDataset(Dataset):
    """Dataset class for generating and storing single pendulum trajectories, written in JAX for scalability."""

    def __init__(self, mode: str = "stormer_verlet"):
        """Initialize the dataset by generating pendulum trajectories."""
        self.mode = mode
        self.data = self.generate_data()

    def simulate_trajectory(self,
        init_state: jnp.ndarray,
        mass: float,
        length: float,
        g: float,
        dt: float,
        trajectory_length: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Simulate a single pendulum trajectory.

        Args:
            init_state: Initial state [theta, p].
            mass: Mass of the pendulum.
            length: Length of the pendulum.
            g: Gravitational acceleration.
            dt: Time step.
            trajectory_length: Number of time steps to simulate.

        Returns:
            Tuple of arrays: (states for the entire trajectory, derivatives for the entire trajectory).
        """
        position_derivative_fn = partial(single_pendulum_position_derivative, mass=mass, length=length, g=g)
        momentum_derivative_fn = partial(single_pendulum_momentum_derivative, mass=mass, length=length, g=g)
        integrator = partial(symplectic_integrator_step, 
                             position_derivative_fn=position_derivative_fn,
                             momentum_derivative_fn=momentum_derivative_fn,
                             integration_method=self.mode,
                             dt=dt)

        def body_fun(state: jnp.ndarray, _: Any) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
            new_state = integrator(state)
            new_state = new_state.at[0].set((new_state[0] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
            theta_dot = position_derivative_fn(new_state)
            p_dot = momentum_derivative_fn(new_state)
            return new_state, (new_state, (theta_dot, p_dot))

        _, (trajectory, derivatives) = jax.lax.scan(body_fun, init_state, jnp.arange(trajectory_length))
        
        # Add noise to match the original implementation
        key = jax.random.PRNGKey(0)
        noise = jax.random.normal(key, shape=trajectory.shape) * config['noise_std']
        noisy_trajectory = trajectory + noise
        
        return noisy_trajectory, derivatives

    def generate_data(self) -> torch.Tensor:
        """
        Generate the dataset of pendulum trajectories including energies.

        Returns:
            Tensor containing all trajectories with their parameters, derivatives, and energies.
        """
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, config['num_trajectories'])
        
        def generate_state(key: jnp.ndarray) -> jnp.ndarray:
            return jax.random.uniform(
                key,
                shape=(2,),
                minval=jnp.array([config['theta_min'], config['p_min']]),
                maxval=jnp.array([config['theta_max'], config['p_max']])
            )
        
        generate_states = jax.vmap(generate_state)
        init_states = generate_states(keys)
    
        # Use fixed parameters to match the original implementation
        masses = jnp.full((config['num_trajectories'],), config['mass'])
        lengths = jnp.full((config['num_trajectories'],), config['length'])
        gs = jnp.full((config['num_trajectories'],), config['g'])

        simulate_trajectory_jit = jit(self.simulate_trajectory, static_argnums=(5,))
        simulate_batch = vmap(simulate_trajectory_jit, in_axes=(0, 0, 0, 0, None, None))
        
        trajectories, derivatives = simulate_batch(
            init_states,
            masses,
            lengths,
            gs,
            config['dt'],
            config['trajectory_length']
        )

        if trajectories.size == 0:
            raise ValueError("Trajectories array is empty")

        mass_length_g = jnp.stack([masses, lengths, gs], axis=-1)
        mass_length_g = jnp.tile(mass_length_g[:, jnp.newaxis, :], (1, config['trajectory_length'], 1))

        theta_dot, p_dot = derivatives
    
        # Calculate energies (we add those to the dataset)
        def calculate_single_pendulum_energies(mass: float,
                                               length: float,
                                               g: float,
                                               thetas: jnp.ndarray,
                                               ps: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """Calculate kinetic, potential, and total energy of the pendulum."""

            kinetic = (0.5 * (ps/length)**2) / mass
            potential = mass * g * length * (1 - jnp.cos(thetas))
            total = kinetic + potential
            return kinetic, potential, total

        calculate_energies_batch = jax.vmap(calculate_single_pendulum_energies, in_axes=(0, 0, 0, 0, 0))
        kinetic, potential, total = calculate_energies_batch(
            masses[:, jnp.newaxis],
            lengths[:, jnp.newaxis],
            gs[:, jnp.newaxis],
            trajectories[..., 0],  # thetas
            trajectories[..., 1]   # ps
        )

        # Reorder the data to [theta, p, mass, length, g, dH_dtheta, dH_dp, kinetic, potential, total]
        data = jnp.concatenate([
            trajectories,
            mass_length_g,
            theta_dot,
            p_dot,
            kinetic[..., jnp.newaxis],
            potential[..., jnp.newaxis],
            total[..., jnp.newaxis]
        ], axis=-1)
        
        return torch.tensor(np.array(data), dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of trajectories in the dataset."""
        return config['num_trajectories']

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single trajectory from the dataset.

        Args:
            idx: Index of the trajectory to retrieve.

        Returns:
            Tensor containing the trajectory, its parameters, and derivatives.
        """
        return self.data[idx]

def save_dataset(dataset: SinglePendulumDataset, filename: str) -> None:
    """
    Save the generated dataset to a file.

    Args:
        dataset: The SinglePendulumDataset to save.
        filename: Name of the file to save the dataset to.
    """
    torch.save(dataset.data, filename)

def generate_single_pendulum_data(
    output_path: Optional[str] = None,
    integration_mode: str = "stormer_verlet"
) -> None:
    """
    Generate and save a single pendulum dataset, create an animation of the first trajectory,
    and print dataset statistics.

    This function creates a SinglePendulumDataset, saves it to a specified or default location,
    generates an animation of the first trajectory, and prints a preview and statistics of the dataset.

    Args:
        output_path (Optional[str], optional): Path to save the generated dataset. 
            If None, a default path will be used. Defaults to None.
        integration_mode (str, optional): Integration method to use for generating the dataset. 
            Defaults to "stormer_verlet".

    Raises:
        Exception: If there's an error during the data generation process.

    Returns:
        None
    """
    try:
        # Generate the dataset
        dataset = SinglePendulumDataset(mode=integration_mode)
        
        # Set default output path if not provided
        if output_path is None:
            output_path = os.path.join("data", "single_pendulum", f"single_pendulum_dataset_{integration_mode}.pt")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the dataset
        save_dataset(dataset, output_path)
        print(f"Single pendulum dataset saved to {output_path}")

        # Generate and save animation of the first trajectory
        first_trajectory = dataset.data[0]
        animation_path = os.path.join("results/simulations", f"single_pendulum_{integration_mode}.gif")
        os.makedirs(os.path.dirname(animation_path), exist_ok=True)
        animate_single_pendulum_with_energy_and_phase(first_trajectory, save_path=animation_path)
        print(f"Animation of the first trajectory saved to {animation_path}")

        # Create a DataFrame for dataset preview and statistics
        df = pd.DataFrame(
            dataset.data.view(-1, dataset.data.size(-1)).numpy(),
            columns=['theta', 'p', 'mass', 'length', 'g', 'theta_dot', 'p_dot', 'kinetic', 'potential', 'total']
        )
        
        # Print dataset preview
        print("\nDataset preview:")
        print(df.head())
        
        # Print dataset statistics
        print("\nDataset statistics:")
        print(df.describe())

    except Exception as e:
        print(f"Error generating single pendulum data: {str(e)}")
        import traceback
        traceback.print_exc()