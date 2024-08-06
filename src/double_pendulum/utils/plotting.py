import os
from typing import Tuple, Dict, List, Callable
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.double_pendulum.config import double_pendulum_config as config

def animate_double_pendulum_with_energy_and_phase(trajectory: torch.Tensor, save_path: str = None):
    """
    Animate a double pendulum trajectory with energy plot and six 2D phase space plots.

    Args:
        trajectory (torch.Tensor): A tensor of shape (time_steps, 16) where each row contains
                                   [theta1, theta2, p1, p2, m1, m2, l1, l2, g, theta1_dot, theta2_dot, p1_dot, p2_dot, kinetic, potential, total].
        save_path (str, optional): If provided, save the animation to this path.

    Returns:
        matplotlib.animation.FuncAnimation: The animation object.
    """
    theta1 = trajectory[:, 0].numpy()
    theta2 = trajectory[:, 1].numpy()
    p1 = trajectory[:, 2].numpy()
    p2 = trajectory[:, 3].numpy()
    m1, m2, l1, l2, g = trajectory[0, 4:9].numpy()
    kinetic = trajectory[:, 13].numpy()
    potential = trajectory[:, 14].numpy()
    total = trajectory[:, 15].numpy()
    
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    time = np.arange(len(theta1)) * config['dt']

    # Set up the figure and axes
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3)
    
    ax_pendulum = fig.add_subplot(gs[0, 0])
    ax_energy = fig.add_subplot(gs[0, 1:])
    
    ax_phase_spaces = [fig.add_subplot(gs[i, j]) for i in range(1, 3) for j in range(3)]
    
    fig.set_tight_layout(True)
    
    # Pendulum subplot
    ax_pendulum.set_xlim(-(l1+l2)*1.1, (l1+l2)*1.1)
    ax_pendulum.set_ylim(-(l1+l2)*1.1, (l1+l2)*1.1)
    ax_pendulum.set_aspect('equal', adjustable='box')
    ax_pendulum.grid(True)
    ax_pendulum.set_title("Double Pendulum")
    ax_pendulum.set_xlabel('x')
    ax_pendulum.set_ylabel('y')

    # Energy subplot
    ax_energy.set_xlim(0, time[-1])
    energy_min = min(min(kinetic), min(potential), min(total))
    energy_max = max(max(kinetic), max(potential), max(total))
    ax_energy.set_ylim(energy_min, energy_max * 1.1)
    ax_energy.set_title("Energy over Time")
    ax_energy.set_xlabel('Time (s)')
    ax_energy.set_ylabel('Energy')
    ax_energy.grid(True)

    # Phase space subplots
    phase_spaces = [
        (theta1, p1, "θ1", "p1"),
        (theta2, p2, "θ2", "p2"),
        (theta1, theta2, "θ1", "θ2"),
        (p1, p2, "p1", "p2"),
        (theta1, p2, "θ1", "p2"),
        (theta2, p1, "θ2", "p1")
    ]

    for ax, (x, y, xlabel, ylabel) in zip(ax_phase_spaces, phase_spaces):
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(min(y), max(y))
        ax.set_title(f"Phase Space ({xlabel} vs {ylabel})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    # Initialize plots
    pendulum_lines, = ax_pendulum.plot([], [], 'o-', lw=2)
    time_text = ax_pendulum.text(0.05, 0.95, '', transform=ax_pendulum.transAxes)
    energy_lines = [
        ax_energy.plot([], [], label=label, lw=2)[0]
        for label in ['Kinetic', 'Potential', 'Total']
    ]
    ax_energy.legend(loc='upper right')
    
    phase_trajectories = [ax.plot([], [], 'b-', alpha=0.5)[0] for ax in ax_phase_spaces]
    phase_points = [ax.plot([], [], 'ro')[0] for ax in ax_phase_spaces]

    def init():
        """Initialize the animation"""
        pendulum_lines.set_data([], [])
        time_text.set_text('')
        for line in energy_lines:
            line.set_data([], [])
        for traj, point in zip(phase_trajectories, phase_points):
            traj.set_data([], [])
            point.set_data([], [])
        return [pendulum_lines, time_text] + energy_lines + phase_trajectories + phase_points

    def animate(i):
        """Update the animation at each frame"""
        pendulum_lines.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
        time_text.set_text(f'Time: {time[i]:.1f}s')
        
        for line, energy in zip(energy_lines, [kinetic, potential, total]):
            line.set_data(time[:i+1], energy[:i+1])
        
        phase_data = [
            (theta1[:i+1], p1[:i+1], theta1[i], p1[i]),
            (theta2[:i+1], p2[:i+1], theta2[i], p2[i]),
            (theta1[:i+1], theta2[:i+1], theta1[i], theta2[i]),
            (p1[:i+1], p2[:i+1], p1[i], p2[i]),
            (theta1[:i+1], p2[:i+1], theta1[i], p2[i]),
            (theta2[:i+1], p1[:i+1], theta2[i], p1[i])
        ]
        
        for traj, point, (x_traj, y_traj, x_point, y_point) in zip(phase_trajectories, phase_points, phase_data):
            traj.set_data(x_traj, y_traj)
            point.set_data([x_point], [y_point])
        
        return [pendulum_lines, time_text] + energy_lines + phase_trajectories + phase_points

    # Create the animation
    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=len(theta1), interval=50, blit=True)

    # Add overall title
    plt.suptitle(f"Double Pendulum Simulation (m1={m1:.2f}, m2={m2:.2f}, l1={l1:.2f}, l2={l2:.2f}, g={g:.2f})")

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

import os
from typing import Callable
import numpy as np
import torch
import matplotlib.pyplot as plt

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

    # Create subplots (4x3 grid)
    fig, axes = plt.subplots(4, 3, figsize=(18, 24))
    fig.suptitle("Phase Vector Field Comparison", fontsize=16)

    # List of all phase space combinations (excluding theta1 vs theta2 and p1 vs p2)
    phase_spaces = [
        (theta1, p1, 0, 2, "θ1", "p1", (-np.pi, np.pi), (-8, 8)),
        (theta2, p2, 1, 3, "θ2", "p2", (-np.pi, np.pi), (-8, 8)),
        (theta1, p2, 0, 3, "θ1", "p2", (-np.pi, np.pi), (-8, 8)),
        (theta2, p1, 1, 2, "θ2", "p1", (-np.pi, np.pi), (-8, 8))
    ]

    for row, (x, y, idx1, idx2, xlabel, ylabel, xlim, ylim) in enumerate(phase_spaces):
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
        for col, (dyn, title) in enumerate(zip([true_dyn, baseline_dyn, hnn_dyn], 
                                               ["True Dynamics", "Baseline Model", "HNN Model"])):
            ax = axes[row, col]
            U = dyn[:, idx1].reshape(X.shape)
            V = dyn[:, idx2].reshape(Y.shape)
            
            # Normalize arrows and increase length by 10%
            norm = np.sqrt(U**2 + V**2)
            max_norm = np.max(norm)
            U = 1.3 * U / max_norm
            V =  1.3 * V / max_norm
            
            ax.quiver(X, Y, U, V, scale=22, alpha=0.7)
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title}: {xlabel} vs {ylabel}")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, 'phase_vector_field_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Phase vector field comparison plot saved in {save_dir}")

def plot_trajectory_comparison(original_trajectory: torch.Tensor,
                               hnn_model: torch.nn.Module,
                               baseline_model: torch.nn.Module,
                               t_span: Tuple[float, float],
                               save_dir: str) -> None:
    """
    Plot and save a comparison of the original trajectory with those generated by HNN and baseline models.

    Args:
        original_trajectory (torch.Tensor): The original trajectory from the dataset.
        hnn_model (torch.nn.Module): The trained HNN model.
        baseline_model (torch.nn.Module): The trained baseline model.
        t_span (Tuple[float, float]): Time span for the trajectory.
        save_dir (str): Directory to save the plot.
    """
    def generate_trajectory(model: torch.nn.Module, x0: torch.Tensor, steps: int) -> torch.Tensor:
        trajectory = [x0]
        model.eval()
        with torch.enable_grad():
            for _ in range(steps - 1):
                x = trajectory[-1].clone().detach().requires_grad_(True)
                dx = model(x.unsqueeze(0)).squeeze()
                x_new = x + dx * (t_span[1] - t_span[0]) / steps
                trajectory.append(x_new)
        return torch.stack(trajectory)

    # Generate trajectories
    x0 = original_trajectory[0]
    steps = len(original_trajectory)
    t = torch.linspace(t_span[0], t_span[1], steps)
    hnn_trajectory = generate_trajectory(hnn_model, x0, steps)
    baseline_trajectory = generate_trajectory(baseline_model, x0, steps)

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Trajectory Comparison in Phase Spaces', fontsize=16)

    # Define phase space combinations
    phase_spaces = [
        (0, 1, "θ1", "θ2"),
        (0, 2, "θ1", "p1"),
        (0, 3, "θ1", "p2"),
        (1, 2, "θ2", "p1"),
        (1, 3, "θ2", "p2"),
        (2, 3, "p1", "p2")
    ]

    for (i, j, xlabel, ylabel), ax in zip(phase_spaces, axes.flatten()):
        ax.plot(original_trajectory[:, i], original_trajectory[:, j], label='Original', linewidth=2)
        ax.plot(hnn_trajectory[:, i].detach(), hnn_trajectory[:, j].detach(), label='HNN', linestyle='--')
        ax.plot(baseline_trajectory[:, i].detach(), baseline_trajectory[:, j].detach(), label='Baseline', linestyle=':')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{xlabel} vs {ylabel}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, 'trajectory_comparison_all_phase_spaces.png'))
    plt.close()

    # Plot individual components over time
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    components = ['θ1', 'θ2', 'p1', 'p2']
    
    for idx, (ax, component) in enumerate(zip(axes.flatten(), components)):
        ax.plot(t, original_trajectory[:, idx], label='Original', linewidth=2)
        ax.plot(t, hnn_trajectory[:, idx].detach(), label='HNN', linestyle='--')
        ax.plot(t, baseline_trajectory[:, idx].detach(), label='Baseline', linestyle=':')
        ax.set_ylabel(component)
        ax.legend()
        ax.grid(True)
        if idx >= 2:
            ax.set_xlabel('Time')
    
    plt.suptitle('Angles and Momenta over Time')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, 'trajectory_comparison_over_time.png'))
    plt.close()

    print(f"Trajectory comparison plots saved in {save_dir}")