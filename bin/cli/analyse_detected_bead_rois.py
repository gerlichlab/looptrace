"""Wrapper around / driver of """

import argparse
import os
import subprocess

from gertils import ExtantFile, ExtantFolder

from looptrace.Drifter import Drifter
from looptrace.ImageHandler import ImageHandler


def workflow(rounds_config: ExtantFile, params_config: ExtantFile, images_folder: ExtantFolder):
    H = ImageHandler(
        rounds_config=rounds_config, 
        params_config=params_config, 
        images_folder=images_folder,
        )
    D = Drifter(image_handler=H)
    rois_path = H.bead_rois_path
    n_pos = D.num_fov
    n_time = H.num_timepoints
    script = os.path.join(os.path.dirname(__file__), "analyse_detected_bead_rois.R")
    cmd_to_run = f"Rscript {script} -i {rois_path} -o {os.path.dirname(rois_path)} --num-fov {n_pos} --num-rounds {n_time}"
    print("Running command: ", cmd_to_run)
    subprocess.check_call(cmd_to_run.split(" "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse detected bead ROIs.")
    parser.add_argument("rounds_config", type=ExtantFile.from_string, help="Imaging rounds config file path")
    parser.add_argument("params_config", type=ExtantFile.from_string, help="Looptrace parameters config file path")
    parser.add_argument("images_folder", type=ExtantFolder.from_string, help="Images folder path")
    args = parser.parse_args()
    workflow(
        rounds_config=args.rounds_config,
        params_config=args.params_config, 
        images_folder=args.images_folder,
        )
