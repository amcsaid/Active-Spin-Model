#!/bin/bash
#
#OAR -l /nodes=1/core=1,walltime=10:00:00
#OAR -q dedicated
#OAR -n SingleSimulation

# Load the necessary modules
module load conda/2021.11-python3.9


# Run your Python script with initial parameters
python3 examples/g_exp_script.py --width 49 --height 49 --density 0.3 --n_steps 20000000 --v0 1.0 --g 2.5