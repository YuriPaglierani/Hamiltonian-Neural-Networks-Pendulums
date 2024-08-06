import jax.numpy as jnp

double_pendulum_config = {
    "mass1": 2.0,
    "mass2": 1.0,
    "length1": 1.5,
    "length2": 2.0,
    "g": 3.0,
    "num_trajectories": 300,
    "trajectory_length": 61,
    "dt": 0.2 * (3./15),
    'theta_min': [-2.0, -2.0],
    'theta_max': [2.0, 2.0],
    'p_min': [-15.0, -7.5],
    'p_max': [15.0, 7.5],
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