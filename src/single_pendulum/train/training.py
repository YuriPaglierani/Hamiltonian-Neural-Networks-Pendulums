import os
from typing import Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm, trange

from src.models.hnn import HNN
from src.single_pendulum.config import single_pendulum_training as cfg
from src.single_pendulum.config import single_pendulum_config as cfg_sp_dynamics
from src.single_pendulum.utils.plotting import plot_losses, plot_trajectory_comparison, plot_phase_vector_field
from src.single_pendulum.data_gen.generator import generate_single_pendulum_data

def get_data(data: torch.Tensor, train_test_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare and split the data into training and test sets.

    Args:
        data (torch.Tensor): The input data tensor.
        train_test_split (float): The fraction of data to use for training.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and test data loaders.
    """
    flattened = data.view(-1, data.size(-1))
    x = flattened[:, :2]  # position and momentum
    dx = flattened[:, 5:7]  # theta_dot and p_dot
    
    split_ix = int(len(x) * train_test_split)
    train_x, test_x = x[:split_ix], x[split_ix:]
    train_dx, test_dx = dx[:split_ix], dx[split_ix:]
    
    train_dataset = TensorDataset(train_x, train_dx)
    test_dataset = TensorDataset(test_x, test_dx)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False)
    
    return train_loader, test_loader

def train(model: nn.Module, 
          optimizer: torch.optim.Optimizer, 
          criterion: nn.Module, 
          train_loader: DataLoader, 
          test_loader: DataLoader, 
          epochs: int = 10) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train the model and return the trained model along with training statistics.

    Args:
        model (nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        criterion (nn.Module): The loss criterion.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for test data.
        epochs (int): Number of epochs to train for.

    Returns:
        Tuple[nn.Module, Dict[str, List[float]]]: Trained model and training statistics.
    """
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    stats = {'train_loss': [], 'test_loss': [], 'learning_rate': []}
    
    # Add learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=cfg['scheduler_factor'], 
                                  patience=cfg['scheduler_patience'], verbose=True)
    
    for epoch in trange(1, epochs + 1, desc="Training"):
        model.train()
        train_losses = []
        for x, dxdt in train_loader:
            x = x.requires_grad_(True)
            dxdt_hat = model(x)
            loss = criterion(dxdt_hat, dxdt)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())
        
        model.eval()
        test_losses = []
        with torch.no_grad():
            for x, dxdt in test_loader:
                dxdt_hat = model(x)
                loss = criterion(dxdt_hat, dxdt)
                test_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        test_loss = np.mean(test_losses)
        stats['train_loss'].append(train_loss)
        stats['test_loss'].append(test_loss)
        stats['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Step the scheduler
        scheduler.step(test_loss)
        
        tqdm.write(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4e}, Test Loss: {test_loss:.4e}, LR: {optimizer.param_groups[0]['lr']:.2e}")

    return model, stats

def train_and_evaluate_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, 
                             model_type: str, images_dir: str, model_dir: str) -> None:
    """
    Train, evaluate, and save results for a given model.

    Args:
        model (nn.Module): The model to train and evaluate.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for test data.
        model_type (str): Type of model (e.g., 'baseline' or 'hnn').
    """
    optimizer = torch.optim.Adam(model.parameters(), cfg['learning_rate'], weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    trained_model, stats = train(model, optimizer, criterion, train_loader, test_loader, epochs=cfg['num_epochs'])

    # Create images directory if it doesn't exist
    os.makedirs(images_dir, exist_ok=True)

    # Save plots
    plot_losses(stats, images_dir, model_type)

    # Save model
    model_path = os.path.join(model_dir, f"model_{model_type}.pth")
    torch.save(trained_model.state_dict(), model_path)
    print(f"{model_type.capitalize()} model saved to {model_path}")
    print(f"Plots saved in {images_dir}")

def train_single_pendulum() -> None:
    """
    Train both baseline and HNN models for the single pendulum experiment.

    Args:
        data_path (str): Path to the single pendulum dataset.
    """
    # Check if dataset exists, if not, generate it
    data_dir = os.path.join("data/", "single_pendulum")

    # if data_dir doesn't contain a file that starts with single_pendulum, and ends with pt, create it 
    if not any([f.startswith('single_pendulum') and f.endswith('.pt') for f in os.listdir(data_dir)]):
        generate_single_pendulum_data()

    # Load the dataset that starts with single_pendulum and ends with pt
    data_path = os.path.join(data_dir, [f for f in os.listdir(data_dir) if f.startswith('single_pendulum') and f.endswith('.pt')][0])
    data = torch.load(data_path)
    train_loader, test_loader = get_data(data, train_test_split=0.8)

    # Extract the first trajectory from the dataset for comparison
    original_trajectory = data[0, :, :2] 

    # Create directories for saving models and plots
    model_dir = os.path.join("results", "single_pendulum", "models")
    images_dir = os.path.join("results", "single_pendulum", "images")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Train and evaluate baseline model
    print("Training Baseline Model...")
    model_baseline = HNN(n_elements=1, hidden_dims=cfg['hidden_dim'], num_layers=cfg['num_layers'], baseline=True)
    train_and_evaluate_model(model_baseline, train_loader, test_loader, 'baseline', images_dir, model_dir)

    # Train and evaluate HNN model
    print("\nTraining HNN Model...")
    model_hnn = HNN(n_elements=1, hidden_dims=cfg['hidden_dim'], num_layers=cfg['num_layers'], baseline=False)
    train_and_evaluate_model(model_hnn, train_loader, test_loader, 'hnn', images_dir, model_dir)

    # Plot trajectory comparison
    t_span = (0, (len(original_trajectory) - 1) * cfg['dt']) 
    plot_trajectory_comparison(original_trajectory, model_hnn, model_baseline, t_span, images_dir)
    print(f"Trajectory comparison plots saved in {images_dir}")

    # Define true dynamics function
    def true_dynamics(X):
        theta, p = X[:, 0], X[:, 1]
        dtheta = p / cfg_sp_dynamics['mass'] / cfg_sp_dynamics['length']**2
        dp = -cfg_sp_dynamics['mass'] * cfg_sp_dynamics['g'] * cfg_sp_dynamics['length'] * np.sin(theta)
        return np.stack([dtheta, dp]).T

    # Plot phase vector field comparison
    plot_phase_vector_field(true_dynamics, model_baseline, model_hnn, images_dir)

    print(f"Training completed. Models saved in {model_dir}")
    print(f"Evaluation plots saved in {images_dir}")

