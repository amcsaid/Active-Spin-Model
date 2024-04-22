"""
This module is the rates manager module. It computes the energy per site, the Hamiltonian, the deltas, and the rates.

"""

from typing import Tuple
import torch
from lvmc.core.lattice import ParticleLattice
import torch.nn.functional as F
from icecream import ic
class EventType(Enum):
    FLIP = 0
    HOP = auto()
    ROTATE = auto()
    ROTATE_NEG = auto()


class RatesManager:
    def __init__(self, lattice: ParticleLattice, **params):
        self.lattice = lattice
        self.params = params
        self.interaction_forces = torch.zeros(lattice.height, lattice.width, 2)
        self.rates = {}
        self.rates_sums = {}
        self.beta = 1.0
        self.v0 = 1.0
        for param, value in params.items():
            setattr(self, param, value)

        self.update_rates()

    def compute_interaction_forces(self) -> torch.Tensor:
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
        self.interaction_forces = F.conv2d(
            padded_particles, kernel, padding=0, groups=2
        ).permute(1, 2, 0)

    def compute_energies(self) -> torch.Tensor:
        """
        Compute the energy per site
        :return: Tensor of shape (lattice.height, lattice.width)
        """
        energies = -torch.sum(self.interaction_forces * self.lattice.particles, dim=2)
        return energies

    @property
    def total_energy(self) -> torch.Tensor:
        """
        Compute the total energy of the lattice
        :return: The total energy of the lattice
        """
        return torch.sum(self.compute_energies())

    def compute_delta(self, event_type: EventType) -> torch.Tensor:
        """
        Compute the change in energy for each event type
        :param event_type: The type of event to compute the change in energy for
        :return: The change in energy for each site for the given event type.
        """
        if isinstance(event_type, int):
            event_type = EventType(event_type)
        if event_type == EventType.ROTATE:
            return self.compute_rotate_delta()

        if event_type == EventType.HOP:
            return self.compute_hop_delta()

        if event_type == EventType.FLIP:
            return -4 * self.energies

        if event_type == EventType.ROTATE_NEG:
            return -self.compute_rotate_delta() - 4 * self.energies

    def compute_rotate_delta(self) -> torch.Tensor:
        """
        Compute the change in energy for a rotation event
        :return: The change in energy for each site for a rotation event
        """
        rotation_matrix = torch.tensor([[0, 1], [-1, 0]], dtype=torch.int8)
        rotated_particles = torch.matmul(self.lattice.particles, rotation_matrix)
        H_ortho = -torch.sum(self.interaction_forces * rotated_particles, dim=2)
        return 2 * (H_ortho - self.energies) + self.occupancy_deltas



if __name__ == "__main__":
    lattice = ParticleLattice(5, 5)
    lattice.populate(0.3)
    ic(lattice)
    rm = RatesManager(lattice)
    rm.compute_unidim_interaction_energies()
