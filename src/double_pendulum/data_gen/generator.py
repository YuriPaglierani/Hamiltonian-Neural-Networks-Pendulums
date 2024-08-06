"""
This module implements a dataset for simulating double pendulum trajectories using JAX.
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

from src.double_pendulum.config import double_pendulum_config as config
from src.double_pendulum.utils.plotting import animate_double_pendulum_with_energy_and_phase
from src.common.utils.integrators import (
    symplectic_integrator_step,
    double_pendulum_position_derivative,
    double_pendulum_momentum_derivative,
)

from functools import partial

jax.config.update("jax_enable_x64", True)

class DoublePendulumDataset(Dataset):
    """Dataset class for generating and storing double pendulum trajectories, written in JAX for scalability."""

    def __init__(self, mode: str = "stormer_verlet"):
        """Initialize the dataset by generating pendulum trajectories."""
        self.mode = mode
        self.data = self.generate_data()

    def simulate_trajectory(self,
        init_state: jnp.ndarray,
        m1: float, # mass first pendulum
        m2: float, # mass second pendulum
        l1: float, # length first pendulum
        l2: float, # length second pendulum
        g: float, # gravity
        dt: float, # time step
        trajectory_length: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Simulate a double pendulum trajectory.

        Args:
            init_state: Initial state [theta1, theta2, p1, p2].
            m1, m2: Masses of the pendulums.
            l1, l2: Lengths of the pendulums.
            g: Gravitational acceleration.
            dt: Time step.
            trajectory_length: Number of time steps to simulate.

        Returns:
            Tuple of arrays: (states for the entire trajectory, derivatives).
        """
        position_derivative_fn = partial(double_pendulum_position_derivative, m1=m1, m2=m2, l1=l1, l2=l2, g=g)
        momentum_derivative_fn = partial(double_pendulum_momentum_derivative, m1=m1, m2=m2, l1=l1, l2=l2, g=g)
        integrator = partial(symplectic_integrator_step, 
                             position_derivative_fn=position_derivative_fn,
                             momentum_derivative_fn=momentum_derivative_fn,
                             integration_method=self.mode,
                             dt=dt)

        def body_fun(state: jnp.ndarray, _: Any) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
            new_state = integrator(state)
            new_state = new_state.at[0].set((new_state[0] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
            new_state = new_state.at[1].set((new_state[1] + jnp.pi) % (2 * jnp.pi) - jnp.pi)
            theta1_dot, theta2_dot = position_derivative_fn(new_state)
            p1_dot, p2_dot = momentum_derivative_fn(new_state)
            return new_state, (new_state, (theta1_dot, theta2_dot, p1_dot, p2_dot))

        _, (trajectory, derivatives) = jax.lax.scan(body_fun, init_state, jnp.arange(trajectory_length))
        
        return trajectory, derivatives

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
                shape=(4,),
                minval=jnp.array([config['theta_min'][0], config['theta_min'][1], config['p_min'][0], config['p_min'][1]]),
                maxval=jnp.array([config['theta_max'][0], config['theta_max'][1], config['p_max'][0], config['p_max'][1]])
            )

        generate_states = jax.vmap(generate_state)
        init_states = generate_states(keys)
    
        m1s = jnp.full((config['num_trajectories'],), config['mass1'])
        m2s = jnp.full((config['num_trajectories'],), config['mass2'])
        l1s = jnp.full((config['num_trajectories'],), config['length1'])
        l2s = jnp.full((config['num_trajectories'],), config['length2'])
        gs = jnp.full((config['num_trajectories'],), config['g'])

        simulate_trajectory_jit = jit(self.simulate_trajectory, static_argnums=(7,))
        simulate_batch = vmap(simulate_trajectory_jit, in_axes=(0, 0, 0, 0, 0, 0, None, None))
        
        trajectories, derivatives = simulate_batch(
            init_states,
            m1s,
            m2s,
            l1s,
            l2s,
            gs,
            config['dt'],
            config['trajectory_length']
        )

        if trajectories.size == 0:
            raise ValueError("Trajectories array is empty")

        params = jnp.stack([m1s, m2s, l1s, l2s, gs], axis=-1)
        params = jnp.tile(params[:, jnp.newaxis, :], (1, config['trajectory_length'], 1))

        # Unpack derivatives and energies
        theta1_dot, theta2_dot, p1_dot, p2_dot = derivatives
    
        def calculate_double_pendulum_energies(m1: float,
                                               m2: float,
                                               l1: float,
                                               l2: float,
                                               g: float,
                                               thetas: jnp.ndarray,
                                               ps: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """Calculate kinetic, potential, and total energy of the pendulum."""

            theta1, theta2 = thetas[..., 0], thetas[..., 1]
            p1, p2 = ps[..., 0], ps[..., 1]

            denom = 2 * m2 * l1**2 * l2**2 * (m1 + m2 * jnp.sin(theta1 - theta2)**2)
            K = (m2 * l2**2 * p1**2 + (m1 + m2) * l1**2 * p2**2 - 2 * m2 * l1 * l2 * jnp.cos(theta1 - theta2) * p1 * p2) / denom
            U = -(m1 + m2) * g * l1 * jnp.cos(theta1) - m2 * g * l2 * jnp.cos(theta2)
            return K, U, K + U
        
        thetas = trajectories[..., :2]
        ps = trajectories[..., 2:]
        
        calculate_energies_batch = jax.vmap(calculate_double_pendulum_energies, in_axes=(0, 0, 0, 0, 0, 0, 0))
        kinetic, potential, total = calculate_energies_batch(
            m1s[:, jnp.newaxis],
            m2s[:, jnp.newaxis],
            l1s[:, jnp.newaxis],
            l2s[:, jnp.newaxis],
            gs[:, jnp.newaxis],
            thetas,
            ps
        )

        # Reorder the data
        data = jnp.concatenate([
            trajectories,
            params,
            theta1_dot[..., jnp.newaxis],
            theta2_dot[..., jnp.newaxis],
            p1_dot[..., jnp.newaxis],
            p2_dot[..., jnp.newaxis],
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
            Tensor containing the trajectory, its parameters, derivatives, and energies.
        """
        return self.data[idx]
    
def save_dataset(dataset: DoublePendulumDataset, filename: str) -> None:
    """
    Save the generated dataset to a file.

    Args:
        dataset: The DoublePendulumDataset to save.
        filename: Name of the file to save the dataset to.
    """
    torch.save(dataset.data, filename)

def generate_double_pendulum_data(
    output_path: Optional[str] = None,
    integration_mode: str = "stormer_verlet"
) -> None:
    """
    Generate and save a double pendulum dataset, create an animation of the first trajectory,
    and print dataset statistics.

    This function creates a DoublePendulumDataset, saves it to a specified or default location,
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
        dataset = DoublePendulumDataset(mode=integration_mode)
        
        # Set default output path if not provided
        if output_path is None:
            output_path = os.path.join("data", "double_pendulum", f"double_pendulum_dataset_{integration_mode}.pt")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the dataset
        save_dataset(dataset, output_path)
        print(f"Double pendulum dataset saved to {output_path}")

        # Generate and save animation of the first trajectory
        first_trajectory = dataset.data[0]
        animation_path = os.path.join("results/simulations", f"double_pendulum_{integration_mode}.gif")
        os.makedirs(os.path.dirname(animation_path), exist_ok=True)
        animate_double_pendulum_with_energy_and_phase(first_trajectory, save_path=animation_path)
        print(f"Animation of the first trajectory saved to {animation_path}")

        # Create a DataFrame for dataset preview and statistics
        df = pd.DataFrame(
            dataset.data.view(-1, dataset.data.size(-1)).numpy(),
            columns=['theta1', 'theta2', 'p1', 'p2', 'm1', 'm2', 'l1', 'l2', 'g', 
                     'theta1_dot', 'theta2_dot', 'p1_dot', 'p2_dot', 'kinetic', 'potential', 'total']
        )
        
        # Print dataset preview
        print("\nDataset preview:")
        print(df.head())
        
        # Print dataset statistics
        print("\nDataset statistics:")
        print(df.describe())

    except Exception as e:
        print(f"Error generating double pendulum data: {str(e)}")
        import traceback
        traceback.print_exc()