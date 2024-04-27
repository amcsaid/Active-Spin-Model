import os
import argparse
from tqdm import tqdm
from rich import print
import torch
import wandb
from lvmc.core.simulation import Simulation
from lvmc.data_handling.data_handler import SimulationDataHandler

def run_simulation(args):
    run = wandb.init(project="hopping_potts_v0", config=args.__dict__)

    # Include 'g' in the directory name
    directory = os.path.join("data", f"g_{args.g}_run_{wandb.run.id}")
    os.makedirs(directory, exist_ok=True)
    
    fname = os.path.join(directory, "simulation_data.hdf5")
    simulation = (
        Simulation(args.g, args.v0)
        .add_lattice(width=args.width, height=args.height)
        .add_particles(density=args.density)
        .build()
    )
    
    handler = SimulationDataHandler(simulation, fname, buffer_limit=20)
    n_particles = simulation.lattice.n_particles
    order_parameter = torch.zeros(args.n_steps)

    for step in tqdm(range(args.n_steps), desc="Simulation Progress"):
        event = simulation.run()
        order_parameter[step] = (
            simulation.lattice.particles.type(torch.float).sum(dim=[0, 1]).norm()
            / n_particles
        )
        wandb.log({"order_parameter": order_parameter[step].item(), "step": step})
        wandb.log({"total_energy": simulation.rm.total_energy, "step": step})
        wandb.log({"time": simulation.t, "step": step})

        handler.collect_event(event)
        if step % 1000 == 0:
            handler.collect_snapshot()

    run.finish()
    handler.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=49, help="Width of the lattice")
    parser.add_argument("--height", type=int, default=49, help="Height of the lattice")
    parser.add_argument("--density", type=float, default=0.3, help="Density of particles")
    parser.add_argument("--n_steps", type=int, default=20000000, help="Number of simulation steps")
    parser.add_argument("--g", type=float, required=True, help="Alignment sensitivity")
    parser.add_argument("--v0", type=float, default=100.0, help="Base migration rate")
    
    args = parser.parse_args()
    run_simulation(args)
