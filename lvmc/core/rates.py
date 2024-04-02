"""
This module is the rates manager module. It computes the energy per site, the Hamiltonian, the deltas, and the rates.

"""

from typing import Tuple
import torch
from lvmc.core.lattice import ParticleLattice
import torch.nn.functional as F
from icecream import ic


class RatesManager:
    def __init__(self, lattice: ParticleLattice, params: dict = {}):
        self.lattice = lattice
        self.params = params

    def compute_unidim_interaction_energies(self) -> torch.Tensor:
        """
        Count the number of neighbours with a specific orientation.
        :return: Tensor of shape (num_orientations, lattice.height, lattice.width)
        """

        # Create a kernel to count the number of neighbours
        kernel = (
            torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # Replicate the kernel for each orientation
        kernel = kernel.repeat(2, 1, 1, 1)
        # Pad the particles tensor to handle boundary conditions
        padded_particles = F.pad(
            self.lattice.particles.permute(2, 0, 1), pad=(1, 1, 1, 1), mode="circular"
        ).float()

        # Perform convolution to count the number of nearest neighbors with each orientation
        self.nearest_neighbours = F.conv2d(
            padded_particles, kernel, padding=0, groups=2
        )
        ic(self.nearest_neighbours)


if __name__ == "__main__":
    lattice = ParticleLattice(5, 5)
    lattice.populate(0.3)
    ic(lattice)
    rm = RatesManager(lattice)
    rm.compute_unidim_interaction_energies()
