
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from looptrace import rouse_polymer
import sys
import os
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SMC loop extrusion simulation and rouse polymer simulation using ensemble state sampling of SMC positions.')
    parser.add_argument("param_path", help="JSON parameters file path")
    parser.add_argument("--no_SMC_sim", help="Use if SMC positions pre-generated in simulation folder.", action='store_true')
    args = parser.parse_args()
    with open(sys.argv[1], "r") as f:
        params = json.load(f)
    if not os.path.isdir(params['out_path']):
        os.makedirs(params['out_path'])
    try:
        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        if not args.no_SMC_sim:
            rouse_polymer.run_SMC_sim_random_halt(params, run_id = array_id)
        rouse_polymer.run_rouse_SMC_sim(params, run_id = array_id)
    except KeyError: 
        rouse_polymer.run_multiple_rouse_sim(params, repeats = params['repeats'])