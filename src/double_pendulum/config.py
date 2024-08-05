import jax.numpy as jnp

double_pendulum_config = {
    "mass1": 2.0,
    "mass2": 1.0,
    "length1": 1.5,
    "length2": 2.0,
    "g": 3.0,
    "num_trajectories": 100,
    "trajectory_length": 120,
    "dt": 0.2 * (3./15),
    'theta_min': [-1.0, -1.0],
    'theta_max': [1.0, 1.0],
    'p_min': [-1, -1],
    'p_max': [1, 1],
}

double_pendulum_training = {
    "dt": 0.2 * (3./15),
    "num_epochs": 400,
    "batch_size": 128,
    "learning_rate": 0.001,
    "hidden_dim": 200,
    "num_layers": 3,
    "scheduler_factor": 0.5,
    "scheduler_patience": 5,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "seed": 42
}