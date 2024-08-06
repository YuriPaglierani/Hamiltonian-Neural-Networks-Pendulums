import os

from typing import Tuple, Dict, List, Callable
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.single_pendulum.config import single_pendulum_config as config

def animate_single_pendulum_with_energy_and_phase(trajectory: torch.Tensor, save_path: str = None):
    """
    Animate a single pendulum trajectory with energy plots and phase space.

    Args:
        trajectory (torch.Tensor): A tensor of shape (time_steps, 10) where each row contains
                                   [theta, p, mass, length, g, dH_dtheta, dH_dp, kinetic, potential, total].
        save_path (str, optional): If provided, save the animation to this path.

    Returns:
        matplotlib.animation.FuncAnimation: The animation object.
    """
    thetas = trajectory[:, 0].numpy()
    ps = trajectory[:, 1].numpy()
    mass, length, g = trajectory[0, 2:5].numpy()
    theta_dot = trajectory[:, 5].numpy()
    p_dot = trajectory[:, 6].numpy()
    kinetic = trajectory[:, 7].numpy()
    potential = trajectory[:, 8].numpy()
    total = trajectory[:, 9].numpy()
    
    x = length * np.sin(thetas)
    y = -length * np.cos(thetas)

    time = np.arange(len(thetas)) * config['dt']

    # Set up the figure and axes
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.set_tight_layout(True)
    
    # Pendulum subplot
    ax1.set_xlim(-length*1.1, length*1.1)
    ax1.set_ylim(-length*1.1, length*1.1)
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True)
    ax1.set_title("Single Pendulum")
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # Energy subplot
    ax2.set_xlim(0, time[-1])
    ax2.set_ylim(0, max(total) * 1.1)
    ax2.set_title("Energy over Time")
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Energy')
    ax2.grid(True)

    # Phase space subplot
    ax3.set_xlim(min(thetas), max(thetas))
    ax3.set_ylim(min(ps), max(ps))
    ax3.set_title("Phase Space")
    ax3.set_xlabel('θ')
    ax3.set_ylabel('p')
    ax3.grid(True)

    # Initialize plots
    pendulum_line, = ax1.plot([], [], 'o-', lw=2)
    time_text = ax1.text(0.05, 0.95, '', transform=ax1.transAxes)
    kinetic_line, = ax2.plot([], [], label='Kinetic')
    potential_line, = ax2.plot([], [], label='Potential')
    total_line, = ax2.plot([], [], label='Total')
    phase_trajectory, = ax3.plot([], [], 'b-', alpha=0.5)
    phase_point, = ax3.plot([], [], 'ro')
    phase_arrow = ax3.quiver([], [], [], [], color='r', scale=1, scale_units='xy', angles='xy')
    
    # Add legend with fixed position
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))

    def init():
        """Initialize the animation"""
        pendulum_line.set_data([], [])
        time_text.set_text('')
        kinetic_line.set_data([], [])
        potential_line.set_data([], [])
        total_line.set_data([], [])
        phase_trajectory.set_data([], [])
        phase_point.set_data([], [])
        phase_arrow.set_UVC([], [])
        return pendulum_line, time_text, kinetic_line, potential_line, total_line, phase_trajectory, phase_point, phase_arrow

    def animate(i):
        """Update the animation at each frame"""
        pendulum_line.set_data([0, x[i]], [0, y[i]])
        time_text.set_text(f'Time: {time[i]:.1f}s')
        kinetic_line.set_data(time[:i+1], kinetic[:i+1])
        potential_line.set_data(time[:i+1], potential[:i+1])
        total_line.set_data(time[:i+1], total[:i+1])
        phase_trajectory.set_data(thetas[:i+1], ps[:i+1])
        phase_point.set_data([thetas[i]], [ps[i]])
        phase_arrow.set_UVC([theta_dot[i]], [p_dot[i]])  # Use pre-calculated derivatives
        phase_arrow.set_offsets([[thetas[i], ps[i]]])
        return pendulum_line, time_text, kinetic_line, potential_line, total_line, phase_trajectory, phase_point, phase_arrow

    # Create the animation
    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=len(thetas), interval=50, blit=True)

    # Add overall title
    plt.suptitle(f"Single Pendulum Simulation (m={mass:.2f}, l={length:.2f}, g={g:.2f})")

    if save_path:
        anim.save(save_path, writer='pillow', fps=20)
        plt.close(fig)
    else:
        plt.show()

    return anim

def plot_losses(stats: Dict[str, List[float]], save_dir: str, model_type: str) -> None:
    """
    Plot and save the training and test loss curves.

    Args:
        stats (Dict[str, List[float]]): Dictionary containing train and test loss histories.
        save_dir (str): Directory to save the plot.
        model_type (str): Type of model (e.g., 'baseline' or 'hnn') for the filename.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(stats['train_loss'], label='Train Loss')
    plt.plot(stats['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title(f'Training and Test Losses - {model_type.capitalize()} Model')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'loss_plot_{model_type}.png'))
    plt.close()

def plot_phase_vector_field(true_dynamics: Callable, baseline_model: torch.nn.Module, 
                            hnn_model: torch.nn.Module, save_dir: str) -> None:
    """
    Plot and save the phase vector field for the true dynamics, baseline model, and HNN model.

    Args:
        true_dynamics (Callable): Function representing the true system dynamics.
        baseline_model (torch.nn.Module): The trained baseline model.
        hnn_model (torch.nn.Module): The trained HNN model.
        save_dir (str): Directory to save the plot.
    """
    # Create a grid of points in phase space
    theta1 = np.linspace(-np.pi, np.pi, 20)
    theta2 = np.linspace(-np.pi, np.pi, 20)
    p1 = np.linspace(-8, 8, 20)
    p2 = np.linspace(-8, 8, 20)

    # Create subplots (2x3 grid for 6 phase space combinations)
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle("Phase Vector Field Comparison", fontsize=16)

    # List of all phase space combinations
    phase_spaces = [
        (theta1, theta2, 0, 1, "θ1", "θ2"),
        (theta1, p1, 0, 2, "θ1", "p1"),
        (theta2, p2, 1, 3, "θ2", "p2"),
        (p1, p2, 2, 3, "p1", "p2"),
        (theta1, p2, 0, 3, "θ1", "p2"),
        (theta2, p1, 1, 2, "θ2", "p1")
    ]

    for idx, (x, y, idx1, idx2, xlabel, ylabel) in enumerate(phase_spaces):
        ax = axes[idx // 3, idx % 3]
        X, Y = np.meshgrid(x, y)
        
        # Prepare the full state space
        states = np.zeros((X.size, 4))
        states[:, idx1] = X.flatten()
        states[:, idx2] = Y.flatten()
        
        # Compute dynamics for all models
        true_dyn = true_dynamics(states)
        
        baseline_model.eval()
        hnn_model.eval()
        with torch.no_grad():
            states_torch = torch.tensor(states, dtype=torch.float32)
            baseline_dyn = baseline_model(states_torch).numpy()
            hnn_dyn = hnn_model(states_torch).numpy()
        
        # Plot for all three models
        for i, (dyn, title, color) in enumerate(zip([true_dyn, baseline_dyn, hnn_dyn], 
                                                    ["True Dynamics", "Baseline Model", "HNN Model"],
                                                    ['r', 'g', 'b'])):
            U = dyn[:, idx1].reshape(X.shape)
            V = dyn[:, idx2].reshape(Y.shape)
            
            # Normalize arrows
            norm = np.sqrt(U**2 + V**2)
            U = U / (norm + 1e-8)  # Add small epsilon to avoid division by zero
            V = V / (norm + 1e-8)
            
            ax.quiver(X, Y, U, V, color=color, scale=50, label=title, alpha=0.7)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Phase Space: {xlabel} vs {ylabel}")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_aspect('equal')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'phase_vector_field_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Phase vector field comparison plot saved in {save_dir}")

def plot_trajectory_comparison(original_trajectory: torch.Tensor,
                               hnn_model: torch.nn.Module,
                               baseline_model: torch.nn.Module,
                               t_span: Tuple[float, float],
                               save_dir: str) -> None:
    """
    Plot and save a comparison of the original trajectory with those generated by HNN and baseline models,
    using the Störmer-Verlet integration technique.

    Args:
        original_trajectory (torch.Tensor): The original trajectory from the dataset.
        hnn_model (torch.nn.Module): The trained HNN model.
        baseline_model (torch.nn.Module): The trained baseline model.
        t_span (Tuple[float, float]): Time span for the trajectory.
        save_dir (str): Directory to save the plot.
    """
    def stormer_verlet_step(model: torch.nn.Module, state: torch.Tensor, dt: float) -> torch.Tensor:
        """Perform a single step of Störmer-Verlet integration."""
        q, p = state[0], state[1]
        with torch.enable_grad():
            state.requires_grad_(True)
            
            # Half step in momentum
            dstate = model(state.unsqueeze(0)).squeeze()
            dq, dp = dstate[0], dstate[1]
            p_half = p + 0.5 * dt * dp
            
            # Full step in position
            q_new = q + dt * dq
            
            # Recompute derivatives at new position
            new_state = torch.tensor([q_new, p_half], requires_grad=True)
            dstate_new = model(new_state.unsqueeze(0)).squeeze()
            dp_new = dstate_new[1]
            
            # Complete step in momentum
            p_new = p_half + 0.5 * dt * dp_new
        
        return torch.tensor([q_new.item(), p_new.item()])

    def generate_trajectory(model: torch.nn.Module, x0: torch.Tensor, steps: int, dt: float) -> torch.Tensor:
        """Generate a trajectory using the Störmer-Verlet method."""
        trajectory = [x0]
        state = x0.clone()
        model.eval()
        for _ in range(steps - 1):
            state = stormer_verlet_step(model, state, dt)
            trajectory.append(state)
        return torch.stack(trajectory)

    # Generate trajectories
    x0 = original_trajectory[0]
    steps = len(original_trajectory)
    dt = (t_span[1] - t_span[0]) / steps
    t = torch.linspace(t_span[0], t_span[1], steps)
    
    hnn_trajectory = generate_trajectory(hnn_model, x0, steps, dt)
    baseline_trajectory = generate_trajectory(baseline_model, x0, steps, dt)

    # Convert tensors to numpy arrays for plotting
    original_np = original_trajectory.detach().numpy()
    hnn_np = hnn_trajectory.detach().numpy()
    baseline_np = baseline_trajectory.detach().numpy()
    t_np = t.numpy()

    # Plot phase space
    plt.figure(figsize=(12, 8))
    plt.plot(original_np[:, 0], original_np[:, 1], label='Original', linewidth=2)
    plt.plot(hnn_np[:, 0], hnn_np[:, 1], label='HNN', linestyle='--')
    plt.plot(baseline_np[:, 0], baseline_np[:, 1], label='Baseline', linestyle=':')
    plt.xlabel('Position (θ)')
    plt.ylabel('Momentum (p)')
    plt.title('Trajectory Comparison in Phase Space')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'trajectory_comparison.png'))
    plt.close()

    # Plot individual components over time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(t_np, original_np[:, 0], label='Original', linewidth=2)
    ax1.plot(t_np, hnn_np[:, 0], label='HNN', linestyle='--')
    ax1.plot(t_np, baseline_np[:, 0], label='Baseline', linestyle=':')
    ax1.set_ylabel('Position (θ)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(t_np, original_np[:, 1], label='Original', linewidth=2)
    ax2.plot(t_np, hnn_np[:, 1], label='HNN', linestyle='--')
    ax2.plot(t_np, baseline_np[:, 1], label='Baseline', linestyle=':')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Momentum (p)')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle('Position and Momentum over Time')
    plt.savefig(os.path.join(save_dir, 'trajectory_comparison_over_time.png'))
    plt.close()

    print(f"Trajectory comparison plots saved in {save_dir}")