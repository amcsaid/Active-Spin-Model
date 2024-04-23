"""
gillepse.py

This file contains the implementation of the Gillepse algorithm (or Kinetic Monte Carlo) for simulating the movement of particles.

Author: EL KHIYATI Zakarya
Date: 12 apr 2024
"""

from typing import Tuple
import torch
from lvmc.core.rates import RatesManager
from lvmc.core.lattice import ParticleLattice, Orientation


class Gillepse:
    def __init__(self, g, v0, seed: Optional[int] = 1337) -> None:
        """
        Initialize the Gillepse algorithm with the given parameters.

        :param g: aligment sensitivity parameter
        :param v0: Hopping to rotation ratio
        """
