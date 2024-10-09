"""Driver for computing all fiducial bead ROIs for a particular imaging experiment"""

import argparse
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
from gertils import ExtantFile, ExtantFolder

from looptrace.Drifter import Drifter
from looptrace.ImageHandler import ImageHandler
from looptrace.bead_roi_generation import generate_all_bead_rois_from_getter

__author__ = "Vince Reuter"


def workflow(
        rounds_config: ExtantFile, 
        params_config: ExtantFile,
        images_folder: ExtantFolder, 
        output_folder: Union[None, Path, ExtantFolder] = None, 
        timepoint_range: Optional[Iterable[int]] = None,
        **joblib_kwargs
        ) -> Iterable[Tuple[Path, pd.DataFrame]]:
    
    # Instantiate the main values needed for this workflow.
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder)
    D = Drifter(image_handler=H)
    
    # Finalise and prepare the output folder.
    output_folder = output_folder or H.bead_rois_path
    output_folder.mkdir(exist_ok=True, parents=False)

    # Determine the range of timepoints / hybridisation rounds to use.
    timepoint_range = timepoint_range or range(H.num_rounds)

    # Function to get (z, y, x) (stack of 2D images) for a particular FOV and imaging round.
    def get_image_stack(pos_idx: int, timepoint_idx: int) -> np.ndarray:
        return D.get_moving_image(pos_idx=pos_idx, timepoint_idx=timepoint_idx)
    
    return generate_all_bead_rois_from_getter(
        get_3d_stack=get_image_stack, 
        iter_position=range(len(D.images_moving)), 
        iter_timepoint=timepoint_range, 
        output_folder=output_folder,
        params=D.bead_roi_parameters,
        **joblib_kwargs
        )


def get_beads_channel(drifter: Drifter) -> int:
    ch_ref = drifter.reference_channel
    ch_mov = drifter.moving_channel
    if ch_mov != ch_ref:
        raise Exception(f"Key for shifted channel differs from key for template channel! ({ch_mov}, {ch_ref})")
    return ch_ref


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver for computing all fiducial bead ROIs for a particular imaging experiment")
    parser.add_argument("rounds_config", type=ExtantFile.from_string, help="Imaging rounds config file path")
    parser.add_argument("params_config", type=ExtantFile.from_string, help="Looptrace parameters config file path")
    parser.add_argument("images_folder", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    
    parser.add_argument("-O", "--output-folder", type=Path, help="Path to folder in which to place output")
    
    parser.add_argument("--prefer-for-joblib", default="threads", help="Argument for joblib.Parallel's 'prefer' parameter")
    parser.add_argument("--num-jobs", type=int, default=-1, help="Argument for joblib.Parallel's n_jobs parameter")
    
    args = parser.parse_args()
    
    workflow(
        rounds_config=args.rounds_config,
        params_config=args.params_config, 
        images_folder=args.images_folder, 
        output_folder=args.output_folder, 
        prefer=args.prefer_for_joblib, 
        n_jobs=args.num_jobs,
        )
