# Single Pendulum Dataset

## Overview
This dataset contains simulated trajectories for a batch of 100 single pendulum systems. It is designed for training and testing Hamiltonian Neural Networks (HNNs) on learning the dynamics of a simple physical system, including energy conservation principles.

## Dataset Specifications
* **Format**: PyTorch tensor saved as a .pt file
* **Size**: Approximately 470.0 kB for 150 trajectories
* **Dimensions**: (num_trajectories, trajectory_length, 10)
* **num_trajectories**: Number of unique pendulum simulations (default: 150)
* **trajectory_length**: Number of time steps in each trajectory (default: 16)
* **10**: Each time step contains [theta, p_theta, mass, length, g, theta_dot, p_dot, kinetic_energy, potential_energy, total_energy]

## Data Components:
0. *theta*: Angular displacement (radians)
1. *p_theta*: Angular momentum
2. *mass*: Mass of the pendulum point (kg)
3. *length*: Length of the pendulum (m)
4. *g*: Gravitational acceleration (m/s^2)
5. *theta_dot*: equal to dH/dp
6. *p_dot*: equal to -dH/dq
7. *kinetic_energy*: Kinetic energy of the pendulum
8. *potential_energy*: Potential energy of the pendulum
9. *total_energy*: Total energy of the pendulum system

## Data Generation
The dataset is generated using a Störmer-Verlet integrator to ensure good stability during the simulation. The initial conditions and pendulum parameters are randomly sampled for each trajectory:
* *Initial angle (theta)*: Uniformly sampled from [config['theta_min'], config['theta_max']]
* *Initial angular momentum (p)*: Uniformly sampled from [config['p_min'], config['p_max']]
* *Mass*: Constant Parameter for all trajectories (default: 1.0)
* *Length*: Constant Parameter for all trajectories (default: 1.0)
* *Gravitational acceleration*: Constant Parameter for all trajectories (default: 3.0 m/s^2)
Energy calculations and system dynamics (theta_dot, p_dot) are computed for each state in the trajectory.

## Usage
This dataset can be used for:
1. Training Hamiltonian Neural Networks to learn the dynamics of single pendulum systems
2. Evaluating the performance of physics-informed machine learning models
3. Studying energy conservation in simulations of simple physical systems
4. Analyzing the relationship between system parameters, state variables, and energy components

When using this dataset, ensure that your model correctly interprets the order of the variables: position (theta), momentum (p_theta), parameters (mass, length, g), dynamics (theta_dot, p_dot), and energy components (kinetic, potential, total).

## Configuration
The dataset generation is controlled by parameters in the `config.py` file. Key parameters include:
* *theta_min, theta_max*: Range for initial angular displacement
* *p_min, p_max*: Range for initial angular momentum
* *mass, length, g*: Shape parameters for gamma distributions
* *num_trajectories*: Number of trajectories to generate
* *trajectory_length*: Number of time steps in each trajectory
* *dt*: Time step size
* *noise_std*: Standard deviation of Gaussian noise added to the dynamics

If you want to modify your simulations, you can do so by adjusting the `config.py` file.

## Notes
* Energy components are included for more comprehensive analysis (with that you can also see how the choice of parameters impacts the final shape of the energies).
