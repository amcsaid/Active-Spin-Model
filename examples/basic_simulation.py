from lvmc.core.simulation import Simulation
from tqdm import tqdm
from utils import *
from parameters import *
from rich import print
import argparse
import torch



def main(g, v0, width, height, density, n_steps=2000):
    #set up the obstacles
    obstacles = torch.zeros((height, width), dtype=torch.bool)
    obstacles[0, :] = True
    obstacles[-1, :] = True
    # Initialize the Simulation
    simulation = (
        Simulation(g, v0)
        .add_lattice(width=width, height=height)
        # .add_obstacles(obstacles)
        .add_particles(density=density)
        .build()
    )


    for _ in range(n_steps):
        event = simulation.run()
        print(simulation.lattice.visualize_lattice())
        print("__"*width)
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=49, help="Width of the lattice")
    parser.add_argument("--height", type=int, default=49, help="Height of the lattice")
    parser.add_argument(
        "--density", type=float, default=0.3, help="Density of particles"
    )
    parser.add_argument("--g", type=float, default=2.0, help="Alignment sensitivity")
    parser.add_argument("--v0", type=float, default=100.0, help="Base migration rate")
    parser.add_argument("--n_steps", type=int, default=2_000_000, help="Number of simulation steps")
    args = parser.parse_args()
    main(args.g, args.v0, args.width, args.height, args.density, args.n_steps)

