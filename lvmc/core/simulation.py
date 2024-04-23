import torch
import numpy as np
from typing import Optional, Tuple
from lvmc.core.lattice import ParticleLattice, Orientation
from lvmc.core.magnetic_field import MagneticField
from lvmc.core.flow import PoiseuilleFlow
from enum import Enum, auto
from typing import NamedTuple, List, Optional
from typing import Tuple
from lvmc.core.rates import RatesManager, EventType
from icecream import ic


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Event(NamedTuple):
    etype: EventType
    x: int
    y: int

    def is_reorientation(self) -> bool:
        return self.etype in {
            EventType.REORIENTATION_UP,
            EventType.REORIENTATION_LEFT,
            EventType.REORIENTATION_DOWN,
            EventType.REORIENTATION_RIGHT,
        }

    def is_migration(self) -> bool:
        return self.etype == EventType.HOP

    def is_rotation(self) -> bool:
        return self.etype == EventType.ROTATE

    def is_birth(self) -> bool:
        return self.etype == EventType.BIRTH

    def is_transport(self) -> bool:
        return self.etype in {
            EventType.TRANSPORT_UP,
            EventType.TRANSPORT_LEFT,
            EventType.TRANSPORT_DOWN,
            EventType.TRANSPORT_RIGHT,
        }


class Simulation:
    def __init__(
        self,
        g: float,
        v0: float,
        seed: Optional[int] = 1337,
    ) -> None:
        """
        Initialize the simulation with base parameters.

        :param g: Alignment sensitivity parameter.
        :param v0: Base transition rate for particle movement.
        """

        self.g = g
        self.v0 = v0
        self.generator = torch.Generator(device=device)
        if seed is not None:
            self.generator.manual_seed(seed)
        # Initialize time
        self.t = 0.0

    def add_lattice(self, width: int, height: int) -> None:
        """
        Add a lattice to the simulation.

        :param width: The width of the lattice.
        :param height: The height of the lattice.
        """
        self.width = width
        self.height = height
        self.lattice = ParticleLattice(width, height, generator=self.generator)
        return self

    def add_control_field(self, direction: int = 0) -> None:
        """
        Add a control field to the simulation.

        :param direction: The direction of the control field.
        """
        self.control_field = MagneticField(direction)
        return self

    def add_flow(self, flow_params: dict) -> None:
        """
        Add a flow to the simulation.

        :param flow_params: A dictionary containing the parameters of the flow.
        """
        if flow_params["type"] == "Poiseuille":
            self.flow = PoiseuilleFlow(
                width=self.width, height=self.height, v1=flow_params["v1"]
            )
            print("Added Poiseuille flow to the simulation.")
        # Increment the number of event types
        return self

    def add_obstacles(self, obstacles: torch.Tensor) -> None:
        """
        Add obstacles to the lattice.

        :param obstacles: A tensor representing the obstacles on the lattice.
        """
        # Check the shape of the obstacles tensor
        assert obstacles.shape == (
            self.height,
            self.width,
        ), "The shape of the obstacles tensor is not correct."
        self.lattice.set_obstacles(obstacles)
        return self

    def add_sinks(self, sinks: torch.Tensor) -> None:
        """
        Add sinks to the lattice.

        :param sinks: A tensor representing the sinks on the lattice.
        """
        # Check the shape of the sinks tensor
        assert sinks.shape == (
            self.height,
            self.width,
        ), "The shape of the sinks tensor is not correct."
        self.lattice.set_sinks(sinks)
        return self

    def add_sources(self, sources: torch.Tensor) -> None:
        """
        Add sources to the lattice.

        :param sources: A tensor representing the sources on the lattice.
        """
        # Check the shape of the sources tensor
        assert sources.shape == (
            self.height,
            self.width,
        ), "The shape of the sources tensor is not correct."
        self.lattice.set_sources(sources)
        return self

    def add_particles(self, density: float) -> None:
        """
        Populate the lattice with particles.

        :param density: The density of the particles.
        """
        n_added = self.lattice.populate(density)
        return self

    def build(self) -> None:
        """
        Build the simulation.
        """
        self.initialize_rates()
        return self

    def add_particle(self, x: int, y: int, orientation: Orientation = None) -> None:
        """
        Add a particle at the specified location.

        :param x: The x-coordinate of the location.
        :param y: The y-coordinate of the location.
        :param orientation: The orientation of the particle.
        """
        self.lattice.add_particle(x, y, orientation)
        self.rm.update_rates()

    def populate_lattice(self, density: float) -> None:
        """
        Populate the lattice with particles.

        :param density: The density of the particles.
        """
        n_added = self.lattice.populate(density)
        self.rm.update_rates()

    def initialize_rates(self) -> None:
        """
        Instantiate the rates manager.
        """
        self.rm = RatesManager(self.lattice, beta=self.g, v0=self.v0)

    def perform_event(self, event: Event) -> List[tuple]:
        """
        Execute the specified event on the lattice.

        This method determines the type of the given event (reorientation or migration)
        and performs the corresponding action on the lattice. In case of a reorientation
        event, it reorients the particle at the specified location. For a migration event,
        it moves the particle to a new location.

        :param event: The event object to be executed. It contains the event type and
                    the coordinates (x, y) on the lattice where the event occurs.
        """

        if event.is_migration() and not self.lattice._is_empty(event.x, event.y):
            new_pos = self.lattice.move_particle(event.x, event.y)
            return [(event.x, event.y)] + new_pos

        elif event.is_rotation() and not self.lattice._is_empty(event.x, event.y):
            self.lattice.rotate(event.x, event.y)

    def run(self) -> Event:
        """
        Run the simulation for a single time step.

        :return: An Optional tuple (event_type, x, y) representing the event, or None.
        """
        event = self.choose_event()
        affected_sites = self.perform_event(event)
        self.rm.update_rates()
        return event

    def sample_site(self, rates: torch.Tensor) -> Tuple[int, int]:
        """
        Sample a site y, x where the event will take place.

        :param: A 2D tensor representing the likelihood of sampling each site.
        :return: a tuple of integers representing the sampled coordinates
        """

        rates_flat = rates.view(-1)
        total_rate = rates_flat.sum().item()

        if total_rate == 0:
            raise ValueError

        random_value = (
            torch.rand(1, device=device, generator=self.generator).item() * total_rate
        )

        cumulative_rates = torch.cumsum(rates_flat, dim=0)
        chosen_index = torch.searchsorted(cumulative_rates, random_value).item()
        y, x = np.unravel_index(chosen_index, rates.shape)

        return y, x

    def sample_event_type(self, event_rates: dict) -> EventType:
        """
        Sample the event type to occur.
        :params: A dictionary.
        :return: The sampled event type.
        """

        total_rate = sum(event_rates.values())
        random_value = (
            torch.rand(1, device=device, generator=self.generator).item() * total_rate
        )
        cumsum = 0
        for etype, rate in event_rates.items():
            cumsum += rate
            if random_value <= cumsum:
                return etype

    def choose_event(self) -> Event:
        """
        Sample the event to perform

        :return: the event to perform
        """
        event_type = self.sample_event_type(self.rm.rates_sums)
        y, x = self.sample_site(self.rm.rates[event_type])
        return Event(event_type, x, y)
