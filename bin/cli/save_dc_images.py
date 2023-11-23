
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from looptrace.ImageHandler import ImageHandler
from looptrace.Drifter import Drifter
import dask.array as da
import napari
import sys
import os

if __name__ == '__main__':
    H = ImageHandler(sys.argv[1])
    try:
        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        array_id = None

    D = Drifter(H, array_id = array_id)
    D.save_coarse_dc_images()