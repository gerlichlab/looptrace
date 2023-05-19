
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from argparse import ArgumentParser
import os
import subprocess
from typing import *


if __name__ == '__main__':
    
    parser = ArgumentParser(description='Run (or prepare a run of) a step of the processing pipeline on a computing cluster')
    
    parser.add_argument("--job_path", help="Folder to save cluster job files.")
    parser.add_argument('--job_name', help='Job name.', default='cluster_job')
    parser.add_argument('--time', help="Time for cluster job.", default = '0-0:30')
    parser.add_argument('--cpus_per_task', help='Number of cores.', default='4')
    parser.add_argument('--mem_per_cpu', help='Memory per CPU core', default='4G')
    parser.add_argument('--array', help="Number of jobs to spawn in an array for cluster job.", default = '0')
    parser.add_argument('--partition', help="Partion for cluster job.", default='htc-el8')
    parser.add_argument('--gres', help="Extra requests (usually GPU) for cluster job.")
    parser.add_argument('--module', help='Name of module to load')

    conda_env = parser.add_mutually_exclusive_group()
    conda_env.add_argument('--conda-env-name', help="Name of conda environment to use (should just be a simple name)")
    conda_env.add_argument('--conda-env-prefix', help="Prefix of conda environment to use (should be a path to the root folder of an extant conda environment)")
    
    parser.add_argument('--bin_path', help='Path to python file to run.')
    parser.add_argument("--config_path", help="Config file path")
    parser.add_argument("--image_path", help="Path to folder with images to read.")
    parser.add_argument("--image_save_path", help="(Optional): Path to folder to save images to.")
    parser.add_argument('--additional_options', help = 'Additional options to script.')
    
    parser.add_argument('--run', action="store_true", help='Run the cluster job in addition to making the sbatch file.')
    
    args = parser.parse_args()


    # Make top level directories
    if not os.path.exists(args.job_path):
        os.makedirs(args.job_path)

    job_file = os.path.join(args.job_path, args.job_name + '.sh')

    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --job-name="+args.job_name+'\n')
        fh.writelines('#SBATCH --output='+job_file+'_%A.%a.out\n')
        fh.writelines('#SBATCH --error='+job_file+'_%A.%a.err\n')
        fh.writelines("#SBATCH --time="+args.time+'\n')
        fh.writelines("#SBATCH --cpus-per-task="+args.cpus_per_task+'\n')
        fh.writelines("#SBATCH --mem-per-cpu="+args.mem_per_cpu+'\n')
        fh.writelines("#SBATCH --array="+args.array+'\n')
        fh.writelines("#SBATCH --partition="+args.partition+'\n')
        if args.gres is not None:
            fh.writelines("#SBATCH --gres="+args.gres+'\n')
        fh.writelines("#SBATCH --mail-type=END,FAIL\n")
        fh.writelines('#SBATCH --mail-user='+os.environ.get('USER')+'.'.join(os.environ.get('HOSTNAME').split('.')[-2:])+'\n\n')
        if args.module is not None:
            fh.writelines('module load '+args.module+'\n')
        fh.writelines("which python3\n")

        if args.conda_env_name:
            env_spec = ["-n", args.conda_env_name]
        elif args.conda_env_prefix:
            env_spec = ["-p", args.conda_env_prefix]
        else:
            # no environment to specify
            env_spec = []
        command_base = ["conda", "run"] + env_spec + ["python3", args.bin_path, args.config_path]

        if args.additional_options is not None:
            command_extras = [args.image_path, args.additional_options]
        elif args.image_path is not None:
            command_extas = [args.image_path]
        else:
            command_extras = []
        fh.writelines(' '.join(command_base + command_extras))
        if args.image_save_path:
            fh.writelines(' '.join([' --image_save_path', args.image_save_path]))
    
    if args.run:
        subprocess.run(['sbatch', job_file])