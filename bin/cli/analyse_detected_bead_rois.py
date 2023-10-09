"""Wrapper around / driver of """

import argparse
import os
import subprocess

from gertils import ExtantFile, ExtantFolder

from looptrace.Drifter import Drifter
from looptrace.ImageHandler import handler_from_cli


def workflow(config_file: ExtantFile, images_folder: ExtantFolder):
    """
    Run the bead detection workflow, driving the R script to count lines in each relevant file and visualise.

    Parameters
    ----------
    config_file : gertils.ExtantFile
        Path to the main looptrace processing configuration file, with which to build a looptrace.ImageHandler 
        to determine the counts of positions (fields of view) and hybridisation rounds/timepoints (frames) 
        to pass to the script to know which files to look at to count records.
    images_folder : gertils.ExtantFolder
        Path to the folder with experiment's imaging files
    """
    H = handler_from_cli(config_file=config_file, images_folder=images_folder)
    D = Drifter(image_handler=H)
    rois_path = H.bead_rois_path
    n_pos = D.num_positions
    n_time = H.num_timepoints
    script = os.path.join(os.dirname(__file__), 'analyse_detected_bead_rois.py')
    cmd_to_run = f"Rscript {script} -i {rois_path} -o --num-positions {n_pos} --num-frames {n_time}"
    print("Running command: ", cmd_to_run)
    subprocess.check_call(cmd_to_run.split(" "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deconvolve image data.')
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    parser.add_argument("images_folder", type=ExtantFolder.from_string, help="Images folder path")
    args = parser.parse_args()
    workflow(config_file=args.config_path, images_folder=args.images_folder)
