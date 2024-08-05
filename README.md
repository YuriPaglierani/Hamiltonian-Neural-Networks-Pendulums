# Hamiltonian Neural Network for Pendulum Systems

This repository contains an implementation of Hamiltonian Neural Networks (HNNs) applied to single and double pendulum systems. The project demonstrates how HNNs can learn and predict the dynamics of these classical mechanical systems while preserving important physical properties like energy conservation.

Before starting install the requirements.txt, then "pip install -e ." in the root directory of the project.

python main.py generate single generates the dataset of single pendulum with an animation of the first generated trajectory.

## Project Overview

The main components of this project are:

1. Data generation for single and double pendulum systems using symplectic integrators
2. Implementation of a Hamiltonian Neural Network
3. Training scripts for both single and double pendulum experiments
4. Visualization tools for comparing true and predicted trajectories, as well as energy conservation

## Repository Structure

```
Hamiltonian-Neural-Networks-Pendulum/
├── LICENSE
├── README.md
├── common
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   └── config.cpython-310.pyc
│   ├── config.py
│   └── utils
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-310.pyc
│       │   └── integrators.cpython-310.pyc
│       └── integrators.py
├── double_pendulum
│   ├── __init__.py
│   ├── data
│   │   ├── README.md
│   │   └── double_pendulum_dataset_stormer_verlet.pt
│   ├── data_gen
│   │   ├── __init__.py
│   │   └── generator.py
│   ├── experiments
│   │   ├── __init__.py
│   │   └── experiment.py
│   └── utils
│       ├── __init__.py
│       └── plotting.py
├── models
│   ├── __init__.py
│   ├── hnn.ipynb
│   └── hnn.py
├── requirements.txt
├── simulations
│   ├── double_pendulum_animation_stormer_verlet.gif
│   └── single_pendulum_animation_stormer_verlet.gif
└── single_pendulum
    ├── __init__.py
    ├── data
    │   ├── README.md
    │   └── single_pendulum_dataset_stormer_verlet.pt
    ├── data_gen
    │   ├── __init__.py
    │   └── generator.py
    ├── experiments
    │   ├── __init__.py
    │   └── experiment.py
    └── utils
        ├── __init__.py
        └── plotting.py
```

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/hnn-pendulum-project.git
   cd Hamiltonian-Neural-Networks-Pendulum
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Experiments

To run the single pendulum experiment:

```
python experiments/single_pendulum_experiment.py
```

To run the double pendulum experiment:

```
python experiments/double_pendulum_experiment.py
```

Each experiment will train an HNN model on the respective pendulum system and generate plots comparing the true and predicted trajectories, as well as the energy conservation properties of the HNN versus the true system.

## Results

After running the experiments, you will find the following output files in the project root directory:

- `single_pendulum_results.png`: Trajectory and energy comparison for the single pendulum system
- `double_pendulum_results.png`: Trajectory and energy comparison for the double pendulum system

These plots demonstrate the ability of the HNN to learn and predict the dynamics of the pendulum systems while preserving energy conservation.

## Contributing

Contributions to this project are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.