import torch

# Parameters for ParticleLattice
width = 50
height =25
density = 0.3

# Simulation parameters
g = 2.0  # Alignment sensitivity
v0 = 1.0  # Base transition rate

# Flow parameters
flow_params = {
    # include relevant parameters here
    "type": "Poiseuille",
    "v1": 100,
}


obstacles = torch.zeros((height, width), dtype=torch.bool)
obstacles[0, :] = True
obstacles[-1, :] = True
