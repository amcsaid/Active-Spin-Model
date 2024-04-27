"""
This module is the rates manager module. It computes the energy per site, the Hamiltonian, the deltas, and the rates.

"""

from typing import Tuple
import torch
from lvmc.core.lattice import ParticleLattice
from enum import Enum, auto


class EventType(Enum):
    FLIP = 0
    HOP = auto()
    ROTATE_CW = auto()
    ROTATE_CCW = auto()


class RatesManager:
    R = torch.tensor([[0, 1], [-1, 0]], dtype=torch.int8)  # Rotation matrix

    def __init__(self, lattice: ParticleLattice, **params):
        self.lattice = lattice
        self.params = params
        self.interaction_forces = torch.zeros(
            lattice.height, lattice.width, 2, dtype=torch.int8
        )
        self.deltas = {}
        self.rates = {}
        self.rates_sums = {}
        self.delta_computers = {
            EventType.ROTATE_CW: self.compute_delta_rotate,
            EventType.HOP: self.compute_delta_hop,
            EventType.FLIP: self.compute_delta_flip,
            EventType.ROTATE_CCW: self.compute_delta_rotate_neg,
        }

        for param, value in params.items():
            setattr(self, param, value)

        self.update_rates()

    def compute_interaction_forces(self) -> torch.Tensor:
        """
        Count the number of neighbours with a specific orientation.
        :return: Tensor of shape (num_orientations, lattice.height, lattice.width)
        """
        delta_e = torch.tensor([1, 0], dtype=torch.int8)
        delta_s = torch.tensor([0, 1], dtype=torch.int8)

        y, x = self._get_meshgrid()

        x_s, y_s = self._compute_new_positions(x, y, delta_s[0], delta_s[1])
        x_e, y_e = self._compute_new_positions(x, y, delta_e[0], delta_e[1])
        x_n, y_n = self._compute_new_positions(x, y, -delta_s[0], -delta_s[1])
        x_w, y_w = self._compute_new_positions(x, y, -delta_e[0], -delta_e[1])

        sigma_s = self._perform_change_of_coordinates(self.lattice.particles, x_s, y_s)
        sigma_e = self._perform_change_of_coordinates(self.lattice.particles, x_e, y_e)
        sigma_n = self._perform_change_of_coordinates(self.lattice.particles, x_n, y_n)
        sigma_w = self._perform_change_of_coordinates(self.lattice.particles, x_w, y_w)

        return sigma_s + sigma_e + sigma_n + sigma_w

    def update_interaction_forces(self) -> None:
        """
        Update the interaction forces for the lattice
        """
        self.interaction_forces = self.compute_interaction_forces()

    @staticmethod
    def _perform_change_of_coordinates(
        tensor: torch.Tensor, new_x: torch.Tensor, new_y: torch.Tensor
    ) -> torch.Tensor:
        return tensor[new_y.type(torch.int), new_x.type(torch.int)]

    def compute_energies(self) -> torch.Tensor:
        """
        Compute the energy per site
        :return: Tensor of shape (lattice.height, lattice.width)
        """
        return self.compute_dot_product(
            self.interaction_forces, self.lattice.particles
        )  # -torch.sum(self.interaction_forces * self.lattice.particles, dim=2)

    @property
    def total_energy(self) -> torch.Tensor:
        """
        Compute the total energy of the lattice
        :return: The total energy of the lattice
        """
        return torch.sum(self.compute_energies())

    def _get_meshgrid(self):
        return torch.meshgrid(
            torch.arange(self.lattice.height, dtype=torch.int8),
            torch.arange(self.lattice.width, dtype=torch.int8),
            indexing="ij",
        )

    def _compute_new_positions(self, x, y, dx, dy):
        return (x + dx) % self.lattice.width, (y + dy) % self.lattice.height

    def apply_translational_transformations(self) -> None:
        """
        Apply translational transformations to the lattice
        """
        y, x = self._get_meshgrid()
        dx, dy = self.lattice.particles[..., 0], self.lattice.particles[..., 1]
        self.forward_x, self.forward_y = self._compute_new_positions(x, y, dx, dy)
        # self.backward_x, self.backward_y = self._compute_new_positions(x, y, -dx, -dy)
        # self.right_x, self.right_y = self._compute_new_positions(x, y, dy, -dx)
        # self.left_x, self.left_y = self._compute_new_positions(x, y, -dy, dx)

    @staticmethod
    def compute_dot_product(a, b):
        return -torch.sum(a * b, dim=-1)

    def compute_deltas(self):
        """
        Compute the change in energy for each event type and store in a dictionary.
        """

        for event_type, compute_method in self.delta_computers.items():
            self.deltas[event_type] = compute_method()

    def compute_delta_rotate(self):
        transformed_particles = torch.tensordot(
            self.lattice.particles,
            RatesManager.R - torch.eye(2, dtype=torch.int8),
            dims=1,
        )
        return self.compute_dot_product(transformed_particles, self.interaction_forces)

    def compute_delta_hop(self):
        new_field = (
            self._perform_change_of_coordinates(
                self.interaction_forces, self.forward_x, self.forward_y
            )
            - self.lattice.particles
        )
        return self.compute_dot_product(
            new_field - self.interaction_forces, self.lattice.particles
        )

    def compute_delta_flip(self):
        return -2 * self.compute_dot_product(
            self.lattice.particles, self.interaction_forces
        )

    def compute_delta_rotate_neg(self):
        transformed_particles = torch.tensordot(
            self.lattice.particles,
            -RatesManager.R - torch.eye(2, dtype=torch.int8),
            dims=1,
        )
        return self.compute_dot_product(transformed_particles, self.interaction_forces)

    def update_volume_exclusion_deltas(self) -> torch.Tensor:
        sigma_new = self.lattice.particles[
            self.forward_y.type(torch.int), self.forward_x.type(torch.int)
        ].type(torch.float)
        sigma_new_norm = torch.norm(sigma_new, dim=-1)
        self.ve_deltas = sigma_new_norm / (sigma_new_norm - 1)

    def update_occupancy_deltas(self) -> torch.Tensor:
        self.occupancy_deltas = 1 / torch.norm(
            self.lattice.particles.type(torch.float), dim=-1
        ) -1 

    def compute_rates(self, beta: float = 1, v0: float = 1) -> None:
        """
        Compute the rates for each event type using the computed deltas.
        :param beta: The inverse temperature
        :param v0: The base transition rate
        """
        # Define the specific computation for each rate based on the event type
        self.rates = {
            event_type: torch.exp(
                -beta * self.deltas[event_type] - self.occupancy_deltas
            )
            for event_type in EventType
        }
        self.rates[EventType.HOP] += 1
        self.rates[EventType.HOP] *= v0 * torch.exp(-self.ve_deltas - self.occupancy_deltas)

    def compute_rates_sums(self) -> None:
        """
        Compute the sum of the rates for each event type
        :return: A dictionary with the sum of the rates for each event type
        """
        self.rates_sums = {
            event_type: torch.sum(self.rates[event_type]) for event_type in EventType
        }

    def update_rates(self) -> None:
        """
        Update the rates for each event type
        """
        self.apply_translational_transformations()
        self.update_volume_exclusion_deltas()
        self.update_occupancy_deltas()
        self.update_interaction_forces()
        self.compute_deltas()
        self.compute_rates()
        self.compute_rates_sums()

    def initialize_rates(self) -> None:
        """
        Initialize the rates for each event type
        """
        self.apply_translational_transformations()
        self.update_volume_exclusion_deltas()
        self.update_occupancy_deltas()
        self.update_interaction_forces()
        self.compute_deltas()
        self.compute_rates()
        self.compute_rates_sums()


if __name__ == "__main__":
    from icecream import ic
    from rich import print
    lattice = ParticleLattice(5, 5).populate(0.3)
    rm = RatesManager(lattice)
    ic(rm.interaction_forces)
    ic(rm.total_energy)
    ic(rm.deltas)
    ic(rm.rates)
    ic(lattice)

