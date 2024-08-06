## Double Pendulum Dataset

### Overview

This dataset contains simulated trajectories for a batch of 300 double pendulum systems. It is designed for training and testing Hamiltonian Neural Networks (HNNs) on learning the dynamics of a more complex physical system compared to the single pendulum, including energy conservation principles and phase space dynamics.

### Dataset Specifications
- **Format**: PyTorch tensor saved as a .pt file
- **Size**: Approximately 6.10 MB for 300 trajectories
- **Dimensions**: (num_trajectories, trajectory_length, 16)
- **num_trajectories**: Number of unique double pendulum simulations (default: 300)
- **trajectory_length**: Number of time steps in each trajectory (default: 61)
- **16**: Each time step contains [theta1, theta2, p1, p2, m1, m2, l1, l2, g, theta1_dot, theta2_dot, p1_dot, p2_dot, K, U, E]

#### Data Components:
0-1. *theta1, theta2*: Angular displacements (radians)
2-3. *p1, p2*: Angular momentum
4-5. *m1, m2*: Masses of the first and second pendulum points (kg)
6-7. *l1, l2*: Lengths of the first and second pendulum rods (m)
8. *g*: Gravitational acceleration (m/s^2)
9-10. *theta1_dot, theta2_dot*: equal to dH/dp_i
11-12. *p1_dot, p2_dot*: equal to -dH/dq_i
13. *K*: Kinetic energy of the system
14. *U*: Potential energy of the system
15. *E*: Total energy of the system

### Data Generation
The dataset is generated using a Störmer-Verlet integrator to ensure good stability during the simulation. The initial conditions and pendulum parameters are randomly sampled for each trajectory:

* *Initial angles (theta1, theta2)*: Uniformly sampled from [config['theta_min'], config['theta_max']]
* *Initial angular momenta (p1, p2)*: Uniformly sampled from [config['p_min'], config['p_max']]
* *Masses (m1, m2)*: Constant Parameter for all trajectories (default: config['mass1'], config['mass2'])
* *Lengths (l1, l2)*: Constant Parameter for all trajectories (default: config['length1'], config['length2'])
* *Gravitational acceleration*: Constant Parameter for all trajectories (default: config['g'])

Hamiltonian derivatives and energy components are computed for each state in the trajectory.

### Usage
This dataset is particularly useful for:
1. Training Hamiltonian Neural Networks to learn the dynamics of double pendulum systems
2. Evaluating the performance of physics-informed machine learning models
3. Studying the behavior of chaotic systems and energy conservation in simulations
4. Analyzing phase space dynamics and energy transfer between pendulum components
5. Investigating the relationship between system parameters, state variables, and energy components

When using this dataset, ensure that your model correctly interprets the order of the variables: positions (theta1, theta2), momenta (p1, p2), parameters (m1, m2, l1, l2, g), dynamics (theta1_dot, theta2_dot, p1_dot, p2_dot), and energy components (total kinetic, total potential, total energy).

### Configuration
The dataset generation is controlled by parameters in the `config.py` file. Key parameters include:
* *theta_min, theta_max*: Range for initial angular displacements
* *p_min, p_max*: Range for initial angular momenta
* *mass1, mass2, length1, length2, g*: Shape parameters for gamma distributions
* *num_trajectories*: Number of trajectories to generate
* *trajectory_length*: Number of time steps in each trajectory
* *dt*: Time step size

If you want to modify your simulations, you can do so by adjusting the `config.py` file.

### Notes
* I included also the energy components for more comprehensive analysis (with that you can also see how the choice of parameters impacts the final shape of the energies).