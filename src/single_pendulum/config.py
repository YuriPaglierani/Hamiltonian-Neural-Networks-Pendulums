single_pendulum_config = {
    "mass": 1.0,
    "length": 1.0,
    "g": 3,
    "num_trajectories": 150,
    "trajectory_length": 16,
    "dt": 0.2 * (3./15),
    'theta_min': -2.0,
    'theta_max': +2.0,
    'p_min': -5.0,
    'p_max': 5.0,
    'noise_std': 0.01
}

single_pendulum_training = {
    "dt": 0.2 * (3./15),
    "num_epochs": 300,
    "batch_size": 128,
    "learning_rate": 0.001,
    "hidden_dim": 200,
    "num_layers": 3,
    "scheduler_factor": 0.9,
    "scheduler_patience": 10,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "seed": 42
}