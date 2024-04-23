"""
This module is the rates manager module. It computes the energy per site, the Hamiltonian, the deltas, and the rates.

"""

from typing import Tuple
import torch
from lvmc.core.lattice import ParticleLattice, Orientation
import torch.nn.functional as F
from icecream import ic
import math
from enum import Enum, auto


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

        # # Create a kernel to count the number of neighbours
        # kernel = (
        #     torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32)
        #     .unsqueeze(0)
        #     .unsqueeze(0)
        # )

        # # Replicate the kernel for each orientation
        # kernel = kernel.repeat(2, 1, 1, 1)
        # # Pad the particles tensor to handle boundary conditions
        # padded_particles = F.pad(
        #     self.lattice.particles.permute(2, 0, 1), pad=(1, 1, 1, 1), mode="circular"
        # ).float()

        # # Perform convolution to count the number of nearest neighbors with each orientation
        # self.interaction_forces = F.conv2d(
        #     padded_particles, kernel, padding=0, groups=2
        # ).permute(1, 2, 0)
        delta_e = torch.tensor([1, 0], dtype=torch.int8)
        delta_s = torch.tensor([0, 1], dtype=torch.int8)

        y, x = torch.meshgrid(
            torch.arange(self.lattice.height),
            torch.arange(self.lattice.width),
            indexing="ij",
        ) 

        x_s = (x + delta_s[0]) % self.lattice.width
        y_s = (y + delta_s[1]) % self.lattice.height

        x_e = (x + delta_e[0]) % self.lattice.width
        y_e = (y + delta_e[1]) % self.lattice.height

        x_n = (x - delta_s[0]) % self.lattice.width
        y_n = (y - delta_s[1]) % self.lattice.height

        x_w = (x - delta_e[0]) % self.lattice.width
        y_w = (y - delta_e[1]) % self.lattice.height

        sigma_s = self.lattice.particles[y_s, x_s]
        sigma_e = self.lattice.particles[y_e, x_e]
        sigma_n = self.lattice.particles[y_n, x_n]
        sigma_w = self.lattice.particles[y_w, x_w]

        self.interaction_forces = sigma_s + sigma_e + sigma_n + sigma_w

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

    def compute_new_positions(self):
        y, x = torch.meshgrid(
            torch.arange(self.lattice.height),
            torch.arange(self.lattice.width),
            indexing="ij",
        )
        self.forward_x = (x + self.lattice.particles[...,0]) % self.lattice.width
        self.forward_y = (y + self.lattice.particles[...,1]) % self.lattice.height

        self.backward_x = (x - self.lattice.particles[...,0]) % self.lattice.width
        self.backward_y = (y - self.lattice.particles[...,1]) % self.lattice.height

        # Rotate 90 degrees. pending to check if it is correct
        self.right_x = (x + self.lattice.particles[...,1]) % self.lattice.width
        self.right_y = (y - self.lattice.particles[...,0]) % self.lattice.height

        self.left_x = (x - self.lattice.particles[...,1]) % self.lattice.width
        self.left_y = (y + self.lattice.particles[...,0]) % self.lattice.height
    
    def check_new_positions(self):
        self.compute_new_positions()
        sigma_forward = self.lattice.particles[self.forward_y, self.forward_x]
        sigma_backward = self.lattice.particles[self.backward_y, self.backward_x]
        sigma_right = self.lattice.particles[self.right_y, self.right_x]
        sigma_left = self.lattice.particles[self.left_y, self.left_x]
        F = sigma_forward + sigma_backward + sigma_right + sigma_left
        return F

    def compute_hop_delta(self) -> torch.Tensor:

        F_new = self.interaction_forces[self.forward_y, self.forward_x]
        H_new = (
            -torch.sum(F_new * self.lattice.particles, dim=2)
            + self.occupancy_deltas
            + self.ve_deltas
        )
        return 2 * (H_new - self.energies) 

    def compute_volume_exclusion_delta(self) -> torch.Tensor:
        sigma_new = self.lattice.particles[self.forward_y, self.forward_x].type(torch.float)
        sigma_new_norm = torch.norm(sigma_new, dim=-1)
        self.ve_deltas = sigma_new_norm / (sigma_new_norm - 1)

    def compute_occupancy_delta(self) -> torch.Tensor:
        self.occupancy_deltas = 1 / torch.norm(
            self.lattice.particles.type(torch.float), dim=-1
        )

    def compute_rates(self) -> None:
        """
        Compute the rates for each event type
        :return: A dictionary with the rates for each event type
        """
        self.rates[EventType.ROTATE] = torch.exp(
            -self.beta * self.compute_delta(EventType.ROTATE)
        )
        self.rates[EventType.HOP] = self.v0 * torch.exp(-self.ve_deltas) + torch.exp(
            -self.beta * self.compute_delta(EventType.HOP)
        )
        self.rates[EventType.FLIP] = torch.exp(
            -self.beta * self.compute_delta(EventType.FLIP)
        )
        self.rates[EventType.ROTATE_NEG] = torch.exp(
            -self.beta * self.compute_delta(EventType.ROTATE_NEG)
        )

    def compute_rates_sums(self) -> None:
        """
        Compute the sum of the rates for each event type
        :return: A dictionary with the sum of the rates for each event type
        """
        self.rates_sums[EventType.ROTATE] = torch.sum(
            self.rates[EventType.ROTATE]
        )
        self.rates_sums[EventType.HOP] = torch.sum(self.rates[EventType.HOP])

    def update_rates(self) -> None:
        """
        Update the rates for each event type
        """
        self.compute_new_positions()
        self.compute_volume_exclusion_delta()
        self.compute_occupancy_delta()
        self.compute_interaction_forces()
        self.energies = self.compute_energies()
        self.compute_rates()
        self.compute_rates_sums()


if __name__ == "__main__":
    lattice = ParticleLattice(5, 5)
    lattice.populate(0.6)
    rm = RatesManager(lattice)


    H = 4000
    W = 300

    small_lattice = ParticleLattice(width=W, height=H)

    small_lattice.populate(0.3)

    sigma = small_lattice.particles


    import time
    rm = RatesManager(lattice)
    # rm.update_rates()
    # ic(rm.rates)
    # ic(rm.rates_sums)
    # ic(rm.total_energy)
    start = time.time()
    f = rm.check_new_positions()
    end = time.time()
    ic(f"new: {end - start}")
    start = time.time()
    rm.compute_interaction_forces()
    end = time.time()
    ic(f"with conv: {end - start}")
    ic(f"are the two tensors equal? {torch.equal(f, rm.interaction_forces)}")
