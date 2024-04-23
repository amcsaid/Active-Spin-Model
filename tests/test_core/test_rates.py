import pytest
from lvmc.core.rates import RatesManager, EventType
from lvmc.core.lattice import ParticleLattice, Orientation
import torch
from icecream import ic


class TestRates:
    @pytest.fixture
    def lattice(self):
        lattice = ParticleLattice(width=3, height=3)
        lattice.add_particle(0, 1, Orientation.RIGHT)
        lattice.add_particle(1, 0, Orientation.LEFT)
        lattice.add_particle(1, 2, Orientation.LEFT)
        lattice.add_particle(2, 1, Orientation.LEFT)
        ic(lattice)
        return lattice

    @pytest.fixture
    def rates_manager(self, lattice):
        return RatesManager(lattice, g=1.0, v0=100.0)

    def test_compute_interaction_forces(self, rates_manager):
        rates_manager.compute_interaction_forces()
        ic(rates_manager.interaction_forces)
        assert rates_manager.interaction_forces.shape == (3, 3, 2)
        assert False

    def test_compute_energies(self, rates_manager):
        H = rates_manager.compute_energies()
        ic(H)
        assert H.shape == (3, 3)
        assert H.sum() == 0
        assert H == torch.zeros_like(H)
