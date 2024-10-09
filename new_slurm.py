import argparse
import subprocess
import sys

def slurm_submit(script):
    """
    Submit the SLURM script using sbatch and return the job ID.
    """
    try:
        # Use a list for the command and pass the script via stdin
        output = subprocess.check_output(["sbatch"], input=script, universal_newlines=True)
        job_id = output.strip().split()[-1]
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.output}", file=sys.stderr)
        sys.exit(1)

def submit(gpu_type, num_gpus, job_name, mem, cpus, time, out_dir):
    """
    Construct and submit the SLURM script with the specified parameters.
    """
    # Define GPU configurations
    gpu_configs = {
        'a100': 'nvidia_a100-sxm4-80gb',
        'h100': 'nvidia_h100_80gb_hbm3',
        'a40': 'nvidia_a40',
        # Add more GPU types here if needed
    }

    gpu_resource = f"gpu:{gpu_configs[gpu_type]}:{num_gpus}"

    # Construct the SLURM script
    script = f"""#!/bin/bash
#SBATCH -p olveczkygpu,gpu,gpu_requeue,serial_requeue
#SBATCH --mem={mem}
#SBATCH -c {cpus}
#SBATCH -N 1
#SBATCH -t {time}
#SBATCH -J {job_name}
#SBATCH --gres={gpu_resource}
#SBATCH -o {out_dir}/%x_%j.out

# Load necessary modules and activate environment
source ~/.bashrc
module load Mambaforge/22.11.1-fasrc01
source activate rl
module load cuda/12.2.0-fasrc01

# Display GPU information
nvidia-smi

# Run the Python script
python3 brax_rodent_run_ppo.py
"""

    print(f"Submitting job with GPU type: {gpu_type}, Number of GPUs: {num_gpus}")
    job_id = slurm_submit(script)
    print(f"Job submitted with ID: {job_id}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Submit a SLURM job with specified GPU type.')
    parser.add_argument('--gpu_type', type=str, choices=['a100', 'h100', 'a40'], default='a100',
                        help='Type of GPU to request (default: a100)')
    parser.add_argument('--num_gpus', type=int, default=2,
                        help='Number of GPUs to request (default: 2)')
    parser.add_argument('--job_name', type=str, default='rodent',
                        help='Name of the SLURM job (default: rodent)')
    parser.add_argument('--mem', type=int, default=32000,
                        help='Memory in MB (default: 32000)')
    parser.add_argument('--cpus', type=int, default=4,
                        help='Number of CPU cores (default: 4)')
    parser.add_argument('--time', type=str, default='0-8:00',
                        help='Time limit for the job (default: 0-8:00)')
    parser.add_argument('--out_dir', type=str, default='slurm/out',
                        help='Path for standard output (default: /slurm/out)')

    args = parser.parse_args()

    submit(
        gpu_type=args.gpu_type,
        num_gpus=args.num_gpus,
        job_name=args.job_name,
        mem=args.mem,
        cpus=args.cpus,
        time=args.time,
        out_dir=args.out_dir,
    )

if __name__ == "__main__":
    main()

