import torch

# Parameters for ParticleLattice
width = 49
height = 49
density = 0.3

# Simulation parameters
g = 2.0  # Alignment sensitivity
v0 = 100.0  # Base transition rate


obstacles = torch.zeros((height, width), dtype=torch.bool)
obstacles[0, :] = True
obstacles[-1, :] = True