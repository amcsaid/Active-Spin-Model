import subprocess
import sys

# List of g values to run simulations on
g_values = [0, 0.1, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5]

# Base command that calls your Python script
base_command = [sys.executable, 'examples/g_exp_script.py', '--width', '49', '--height', '49', 
                '--density', '0.3', '--n_steps', '20000000', '--v0', '10.0']

# Start all simulations in parallel
processes = []
for g in g_values:
    log_file = open(f'log_g_{g}.txt', 'w')  # Open a log file for each process
    command = base_command + ['--g', str(g)]
    process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
    processes.append((process, log_file))

# Wait for all processes to complete
for process, log_file in processes:
    process.wait()  # Ensure the process has finished
    log_file.close()  # Close the file after completion
