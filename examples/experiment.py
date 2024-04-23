from lvmc.core.simulation import Simulation
from tqdm import tqdm
from utils import *
from rich import print
import argparse
import torch
import wandb


def run_simulation(args):
    run = wandb.init(project="hopping_potts_v0")

    # Parse command line arguments
    width = args.width
    height = args.height
    density = args.density
    n_steps = args.n_steps
    # Parameters for ParticleLattice
    g = args.g
    v0 = args.v0
    # Initialize the Simulation
    simulation = (
        Simulation(g, v0)
        .add_lattice(width=width, height=height)
        .add_particles(density=density)
        .build()
    )

    n_particles = simulation.lattice.n_particles
    # Initialize weights and biases
    order_parameter = torch.zeros(n_steps)
    # Run the simulation
    for step in tqdm(range(n_steps)):
        event = simulation.run()
        # Calculate and store metrics
        order_parameter[step] = (
            simulation.lattice.particles.type(torch.float).sum(dim=[0, 1]).norm()
            / n_particles
        )
        # Initialize wandb run
        wandb.log({"order_parameter": order_parameter[step]})
        wandb.log({"total energy": simulation.rm.total_energy})

    # Finish wandb run
    run.finish()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=49, help="Width of the lattice")
    parser.add_argument("--height", type=int, default=49, help="Height of the lattice")
    parser.add_argument(
        "--density", type=float, default=0.3, help="Density of particles"
    )
    parser.add_argument(
        "--n_steps", type=int, default=20_000_000, help="Number of simulation steps"
    )
    parser.add_argument("--g", type=float, default=2.0, help="Alignment sensitivity")
    parser.add_argument("--v0", type=float, default=100.0, help="Base migration rate")
    args = parser.parse_args()
    run_simulation(args)
