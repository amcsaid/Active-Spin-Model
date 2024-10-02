# Active Spin Model 

This repository contains the implementation of an active spin model for interacting self-propelled particles. The project focuses on simulating and studying collective behavior in complex systems, combining concepts from statistical physics and active matter.

## Project Structure

```
.
├── code_structure.txt
├── .github
│   └── workflows
│       └── python-app.yml
├── .gitignore
├── LICENSE
├── README.md
├── control_mechanism_and_tasks.png
├── example_animation.gif
├── examples
│   ├── Starter_Notebook.ipynb
│   ├── basic_simulation.py
│   ├── experiment.py
│   ├── g_exp_script.py
│   ├── list_of_g_script.py
│   ├── parameters.py
│   ├── profile_core.py
│   └── utils.py
├── lvmc
│   ├── __init__.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── control_field.py
│   │   ├── flow.py
│   │   ├── lattice.py
│   │   ├── rates.py
│   │   └── simulation.py
│   └── data_handling
│       ├── __init__.py
│       ├── data_collector.py
│       └── data_handler.py
├── requirements.txt
├── run_sim.sh
├── setup.py
└── tests
    ├── __init__.py
    ├── test_core
    │   ├── test_flow.py
    │   ├── test_lattice.py
    │   ├── test_magnetic_field.py
    │   ├── test_rates.py
    │   └── test_simulation.py
    └── test_data_handling
        ├── test_data_collection.py
        └── test_data_export.py
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/zakaryael/active-spin-model.git
   cd active-spin-model
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install the package in editable mode:
   ```
   pip install -e .
   ```

## Usage

### Running a Basic Simulation

To run a basic simulation of the active spin model:

```python
from lvmc.core.simulation import Simulation
from lvmc.core.lattice import Orientation

# Create a simulation
sim = (Simulation(g=1.5, v0=100, seed=42)
       .add_lattice(width=32, height=24))

# Add particles and build the simulation
sim.add_particles(density=0.3).build()

# Run the simulation for a number of steps
for _ in range(1000):
    sim.run()
```

For more detailed examples, refer to the `examples` directory, particularly the `Starter_Notebook.ipynb`.

## Reinforcement Learning Control

The reinforcement learning control part of this project is implemented in a separate repository. For information on how to use reinforcement learning to control the active spin model, please refer to:

[https://github.com/zakaryael/Active-Spin-Gym](https://github.com/zakaryael/Active-Spin-Gym)

This separate repository contains the necessary code and instructions for training and using reinforcement learning agents with the active spin model.

## Contributing

Contributions to this project are welcome!

## Contact

For any questions or concerns, please open an issue in this repository or contact zakarya.el-khiyati@inria.fr
