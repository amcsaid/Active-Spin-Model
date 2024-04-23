from lvmc.core.simulation import Simulation
from tqdm import tqdm
from utils import *
from parameters import *
from rich import print
import argparse
import torch


def main():
    # Initialize the Simulation
    simulation = (
        Simulation(g, v0)
        .add_lattice(width=width, height=height)
        # .add_flow(flow_params)
        # .add_obstacles(obstacles)
        .add_particles(density=density)
        .build()
    )

    n_steps = int(1e5)  # Number of steps to run the simulation for

    for _ in range(n_steps):
        event = simulation.run()
        print(simulation.lattice.visualize_lattice())


if __name__ == "__main__":
    main()
