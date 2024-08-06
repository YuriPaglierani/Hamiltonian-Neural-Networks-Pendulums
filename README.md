# Hamiltonian Neural Networks for Pendulum Systems

This repository contains an implementation of Hamiltonian Neural Networks (HNNs) applied to single and double pendulum systems. The project demonstrates how HNNs can learn and predict the dynamics of these classical mechanical systems, the double pendulum experiment shows also the behaviors of models in chaotic systems.

This work is based on the paper Greydanus, Samuel, Misko Dzamba, and Jason Yosinski. *Hamiltonian Neural Networks*. 2019. [arXiv:1906.01563](https://arxiv.org/abs/1906.01563).

The idea of this project came from the Steve Brunton video about HNNs: [https://www.youtube.com/watch?v=AEOcss20nDA](https://www.youtube.com/watch?v=AEOcss20nDA)

## Project Overview

The main components of this project are:

1. Data generation for single and double pendulum systems using symplectic integrators
2. Implementation of a Hamiltonian Neural Network
3. Training scripts for both single and double pendulum experiments
4. Visualization tools for comparing true and predicted trajectories, as well as energy conservation

## Repository Structure

```
.
├── LICENSE
├── README.md
├── data
│   ├── double_pendulum
│   │   ├── README.md
│   │   └── double_pendulum_dataset_stormer_verlet.pt
│   └── single_pendulum
│       ├── README.md
│       └── single_pendulum_dataset_stormer_verlet.pt
├── main.py
├── notebooks
│   ├── double_pendulum_nb.ipynb
│   └── single_pendulum_nb.ipynb
├── report
├── requirements.txt
├── results
│   ├── double_pendulum
│   │   ├── images
│   │   └── models
│   ├── simulations
│   └── single_pendulum
│       ├── images
│       └── models
└── src
    ├── common
    │   └── utils
    ├── double_pendulum
    │   ├── config.py
    │   ├── data_gen
    │   ├── train
    │   └── utils
    ├── models
    │   ├── hnn.py
    │   └── utils.py
    └── single_pendulum
        ├── config.py
        ├── data_gen
        ├── train
        └── utils
```

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/hamiltonian-neural-networks-pendulum.git
   cd hamiltonian-neural-networks-pendulum
   ```

2. Create a virtual environment and activate it:
   ```
   conda create --name hnn-pendulum python=3.9
   conda activate hnn-pendulum 
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

The project can be used from the command line with the following commands:

### Generate Data

To generate pendulum data:

```
python main.py generate [single|double]
```

- `single` or `double`: Specify the type of pendulum system
- 
Example:
```
python main.py generate single
```

### Train Models

To train pendulum models:

```
python main.py train [single|double]
```

- `single` or `double`: Specify the type of pendulum system to train on

Example:
```
python main.py train double
```

## Results

After running the experiments, you will find the following output files in the `results` directory:

- `results/single_pendulum/images/`: Contains plots for the single pendulum system
- `results/double_pendulum/images/`: Contains plots for the double pendulum system
- `results/simulations/`: Contains animations of the pendulum systems

These plots and animations demonstrate the ability of the HNN to learn and predict the dynamics of the pendulum systems while preserving energy conservation.

## Contributing

Contributions to this project are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
