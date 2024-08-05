import sys
import os
from typing import Tuple, Dict, List, Callable
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.double_pendulum.config import double_pendulum_config as config

def animate_double_pendulum_with_energy_and_phase(trajectory: torch.Tensor, save_path: str = None):
    """
    Animate a double pendulum trajectory with energy plots and phase space.

    Args:
        trajectory (torch.Tensor): A tensor of shape (time_steps, 16) where each row contains
                                   [theta1, theta2, p1, p2, m1, m2, l1, l2, g, dH_dtheta1, dH_dtheta2, dH_dp1, dH_dp2, kinetic, potential, total].
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
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.set_tight_layout(True)
    
    # Pendulum subplot
    ax1.set_xlim(-(l1+l2)*1.1, (l1+l2)*1.1)
    ax1.set_ylim(-(l1+l2)*1.1, (l1+l2)*1.1)
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True)
    ax1.set_title("Double Pendulum")
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # Energy subplot
    ax2.set_xlim(0, time[-1])
    ax2.set_ylim(0, max(total) * 1.1)
    ax2.set_title("Energy over Time")
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Energy')
    ax2.grid(True)

    # Phase space subplot (we'll use theta1 vs theta2)
    ax3.set_xlim(min(theta1), max(theta1))
    ax3.set_ylim(min(theta2), max(theta2))
    ax3.set_title("Phase Space (θ1 vs θ2)")
    ax3.set_xlabel('θ1')
    ax3.set_ylabel('θ2')
    ax3.grid(True)

    # Initialize plots
    pendulum_lines, = ax1.plot([], [], 'o-', lw=2)
    time_text = ax1.text(0.05, 0.95, '', transform=ax1.transAxes)
    kinetic_line, = ax2.plot([], [], label='Kinetic')
    potential_line, = ax2.plot([], [], label='Potential')
    total_line, = ax2.plot([], [], label='Total')
    phase_trajectory, = ax3.plot([], [], 'b-', alpha=0.5)
    phase_point, = ax3.plot([], [], 'ro')
    
    # Add legend with fixed position
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))

    def init():
        """Initialize the animation"""
        pendulum_lines.set_data([], [])
        time_text.set_text('')
        kinetic_line.set_data([], [])
        potential_line.set_data([], [])
        total_line.set_data([], [])
        phase_trajectory.set_data([], [])
        phase_point.set_data([], [])
        return pendulum_lines, time_text, kinetic_line, potential_line, total_line, phase_trajectory, phase_point

    def animate(i):
        """Update the animation at each frame"""
        pendulum_lines.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
        time_text.set_text(f'Time: {time[i]:.1f}s')
        kinetic_line.set_data(time[:i+1], kinetic[:i+1])
        potential_line.set_data(time[:i+1], potential[:i+1])
        total_line.set_data(time[:i+1], total[:i+1])
        phase_trajectory.set_data(theta1[:i+1], theta2[:i+1])
        phase_point.set_data([theta1[i]], [theta2[i]])
        return pendulum_lines, time_text, kinetic_line, potential_line, total_line, phase_trajectory, phase_point

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
    # Create a grid of points in phase space (we'll use theta1 and theta2)
    theta1 = np.linspace(-np.pi, np.pi, 20)
    theta2 = np.linspace(-np.pi, np.pi, 20)
    theta1_mesh, theta2_mesh = np.meshgrid(theta1, theta2)

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Phase Vector Field Comparison (θ1 vs θ2)", fontsize=16)

    # Increase the scale value to reduce arrow lengths
    scale = 50

    # Plot for true dynamics
    X = np.stack([theta1_mesh.flatten(), theta2_mesh.flatten(), np.zeros_like(theta1_mesh.flatten()), np.zeros_like(theta2_mesh.flatten())]).T
    dX = true_dynamics(X)
    ax1.quiver(theta1_mesh, theta2_mesh, dX[:, 0].reshape(theta1_mesh.shape), 
               dX[:, 1].reshape(theta2_mesh.shape), scale=scale)
    ax1.set_title("True Dynamics")
    ax1.set_xlabel("θ1")
    ax1.set_ylabel("θ2")

    # Plot for baseline model
    baseline_model.eval()
    with torch.no_grad():
        X_torch = torch.tensor(X, dtype=torch.float32)
        dX_baseline = baseline_model(X_torch).numpy()
    ax2.quiver(theta1_mesh, theta2_mesh, dX_baseline[:, 0].reshape(theta1_mesh.shape), 
               dX_baseline[:, 1].reshape(theta2_mesh.shape), scale=scale)
    ax2.set_title("Baseline Model")
    ax2.set_xlabel("θ1")
    ax2.set_ylabel("θ2")

    # Plot for HNN model
    hnn_model.eval()
    with torch.no_grad():
        dX_hnn = hnn_model(X_torch).numpy()
    ax3.quiver(theta1_mesh, theta2_mesh, dX_hnn[:, 0].reshape(theta1_mesh.shape), 
               dX_hnn[:, 1].reshape(theta2_mesh.shape), scale=scale)
    ax3.set_title("HNN Model")
    ax3.set_xlabel("θ1")
    ax3.set_ylabel("θ2")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'phase_vector_field_comparison.png'))
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

    # Plot phase space (θ1 vs θ2)
    plt.figure(figsize=(12, 8))
    plt.plot(original_trajectory[:, 0], original_trajectory[:, 1], label='Original', linewidth=2)
    plt.plot(hnn_trajectory[:, 0].detach(), hnn_trajectory[:, 1].detach(), label='HNN', linestyle='--')
    plt.plot(baseline_trajectory[:, 0].detach(), baseline_trajectory[:, 1].detach(), label='Baseline', linestyle=':')
    plt.xlabel('θ1')
    plt.ylabel('θ2')
    plt.title('Trajectory Comparison in Phase Space (θ1 vs θ2)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'trajectory_comparison.png'))
    plt.close()

    # Plot individual components over time
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    
    ax1.plot(t, original_trajectory[:, 0], label='Original', linewidth=2)
    ax1.plot(t, hnn_trajectory[:, 0].detach(), label='HNN', linestyle='--')
    ax1.plot(t, baseline_trajectory[:, 0].detach(), label='Baseline', linestyle=':')
    ax1.set_ylabel('θ1')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(t, original_trajectory[:, 1], label='Original', linewidth=2)
    ax2.plot(t, hnn_trajectory[:, 1].detach(), label='HNN', linestyle='--')
    ax2.plot(t, baseline_trajectory[:, 1].detach(), label='Baseline', linestyle=':')
    ax2.set_ylabel('θ2')
    ax2.legend()
    ax2.grid(True)
    
    ax3.plot(t, original_trajectory[:, 2], label='Original', linewidth=2)
    ax3.plot(t, hnn_trajectory[:, 2].detach(), label='HNN', linestyle='--')
    ax3.plot(t, baseline_trajectory[:, 2].detach(), label='Baseline', linestyle=':')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('p1')
    ax3.legend()
    ax3.grid(True)
    
    ax4.plot(t, original_trajectory[:, 3], label='Original', linewidth=2)
    ax4.plot(t, hnn_trajectory[:, 3].detach(), label='HNN', linestyle='--')
    ax4.plot(t, baseline_trajectory[:, 3].detach(), label='Baseline', linestyle=':')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('p2')
    ax4.legend()
    ax4.grid(True)
    
    plt.suptitle('Angles and Momenta over Time')
    plt.savefig(os.path.join(save_dir, 'trajectory_comparison_over_time.png'))
    plt.close()