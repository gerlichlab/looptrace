"""Driver for computing all fiducial bead ROIs for a particular imaging experiment"""

import argparse
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
from gertils import ExtantFile, ExtantFolder

from looptrace.Drifter import Drifter
from looptrace.ImageHandler import ImageHandler, handler_from_cli
from looptrace.bead_roi_generation import generate_all_bead_rois_from_getter

__author__ = "Vince Reuter"


def get_bead_rois_path(handler: ImageHandler):
    return Path(handler.analysis_path) / "bead_rois__ALL"


def workflow(
        config_file: ExtantFile, 
        images_folder: ExtantFolder, 
        output_folder: Union[None, Path, ExtantFolder] = None, 
        frame_range: Optional[Iterable[int]] = None,
        **joblib_kwargs
        ) -> Iterable[Tuple[Path, pd.DataFrame]]:
    
    # Instantiate the main values needed for this workflow.
    H = handler_from_cli(config_file=config_file, images_folder=images_folder)
    D = Drifter(image_handler=H)
    
    # Finalise and prepare the output folder.
    output_folder = output_folder or get_bead_rois_path(handler=H)
    output_folder.mkdir(exist_ok=True, parents=False)

    # Determine the range of frames / hybridisation rounds to use.
    frame_range = frame_range or range(H.num_frames)

    # Function to get (z, y, x) (stack of 2D images) for a particular FOV and imaging round.
    def get_image_stack(pos_idx: int, frame_idx: int) -> np.ndarray:
        return D.get_moving_image(pos_idx=pos_idx, frame_idx=frame_idx)
    
    return generate_all_bead_rois_from_getter(
        get_3d_stack=get_image_stack, 
        iter_position=range(len(D.images_moving)), 
        iter_frame=frame_range, 
        output_folder=output_folder,
        params=D.get_bead_roi_parameters,
        **joblib_kwargs
        )


def get_images_key(handler: ImageHandler) -> str:
    key_ref = handler.reg_input_template
    key_mov = handler.reg_input_moving
    if key_mov != key_ref:
        raise Exception(f"Key for shifted images differs from key for template images! ({key_mov}, {key_ref})")
    return key_ref


def get_beads_channel(drifter: Drifter) -> int:
    ch_ref = drifter.reference_channel
    ch_mov = drifter.moving_channel
    if ch_mov != ch_ref:
        raise Exception(f"Key for shifted channel differs from key for template channel! ({ch_mov}, {ch_ref})")
    return ch_ref


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver for computing all fiducial bead ROIs for a particular imaging experiment")
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    
    parser.add_argument("-O", "--output-folder", type=Path, help="Path to folder in which to place output")
    
    parser.add_argument("--prefer-for-joblib", default="threads", help="Argument for joblib.Parallel's 'prefer' parameter")
    parser.add_argument("--num-jobs", type=int, default=-1, help="Argument for joblib.Parallel's n_jobs parameter")
    
    args = parser.parse_args()
    
    workflow(
        config_file=args.config_path, 
        images_folder=args.image_path, 
        output_folder=args.output_folder, 
        prefer=args.prefer_for_joblib, 
        n_jobs=args.num_jobs,
        )
