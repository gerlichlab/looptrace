
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

if __name__ == '__main__':
    with open(sys.argv[1], "r") as f:
        params = json.load(f)
    if not os.path.isdir(params['out_path']):
        os.makedirs(params['out_path'])
    rouse_polymer.run_SMC_sim(params, run_id = 0)