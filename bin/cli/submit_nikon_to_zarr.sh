#!/bin/bash
#SBATCH -N 1                        # number of nodes
#SBATCH -n 16                       # number of cores
#SBATCH --mem 16G                    # memory pool for all cores
#SBATCH -t 0-2:00                   # runtime limit (D-HH:MM:SS)
#SBATCH -o slurm.%N.%j.out          # STDOUT
#SBATCH -e slurm.%N.%j.err          # STDERR
#SBATCH --mail-type=END,FAIL        # notifications for job done & fail
#SBATCH --mail-user=kbeckwit@embl.de # send-to address

module load Miniconda3/4.5.12
source /g/easybuild/x86_64/CentOS/7/haswell/software/Miniconda3/4.5.12/bin/activate /home/kbeckwit/env/looptrace
which python3
python3 /home/kbeckwit/git/looptrace_dev/bin/cli/nikon_tiff_to_zarr.py /scratch/kbeckwit/2021-11-09_MT002_2C_4x10_nikon/images/ /scratch/kbeckwit/2021-11-09_MT002_2C_4x10_nikon/zarr_images
deactivate