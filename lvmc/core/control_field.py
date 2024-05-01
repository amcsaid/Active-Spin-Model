from lvmc.core.lattice import ParticleLattice, Orientation  # for type hinting
import numpy as np



class ControlField:
    """
    Class for managing the control field that affects the particles.
    """

    def apply(self, lattice: ParticleLattice) -> None:
        """
        Apply the control field to the lattice.

        :param lattice: The lattice object.
        :type lattice: ParticleLattice
        """
        pass

    def get_state(self) -> np.ndarray:
        """
        Get the state of the control field.

        :return: np.ndarray - The state of the control field.
        """
        pass
class MagneticField(ControlField):
    """
    Class for managing a global magnetic field effects on particles.
    """

    def __init__(self, initial_direction: int = 0):
        """
        Initialize MagneticField with an initial direction.

        :param initial_direction: One of -1, 0, or 1
            - -1: Clockwise
            - 0: None
            - 1: Counterclockwise
        :type initial_direction: int
        """
        self.current_direction = initial_direction

    def update(self, direction: int) -> None:
        """
        Set the current direction of the magnetic field.

        :param direction: One of -1, 0, or 1
        :type direction: int
        """
        self.current_direction = direction

    def apply(self, lattice: ParticleLattice) -> None:
        """
        Apply the magnetic field to all particles on the lattice (a 90 degrees rotation in the prescribed direction).
        :param lattice: The lattice object.
        :type lattice: ParticleLattice
        """
        if self.current_direction == 0:
            pass
        else:
            cw = True if self.current_direction == -1 else False
            lattice.rotate_particles(cw)

    def get_state(self) -> np.ndarray:
        """
        Get the state of the magnetic field.

        :return: np.ndarray - The state of the magnetic field.
        """
        return np.array([self.current_direction])
    
    def get_direction(self) -> int:
        """
        Get the current direction of the magnetic field.

        :return: int - The current direction of the magnetic field.
        """
        return self.current_direction
