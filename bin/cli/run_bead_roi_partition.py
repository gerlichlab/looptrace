"""Bead ROI partitioning"""

import argparse
from pathlib import Path
import subprocess
from typing import *

from gertils import ExtantFile, ExtantFolder
from looptrace import LOOPTRACE_JAR_PATH, LOOPTRACE_JAVA_PACKAGE
from looptrace.ImageHandler import handler_from_cli

__author__ = "Vince Reuter"


def workflow(
    config_file: ExtantFile, 
    images_folder: ExtantFolder, 
    output_folder: Union[None, Path, ExtantFolder] = None,
    ):
    H = handler_from_cli(config_file=config_file, images_folder=images_folder)
    prog_path = f"{LOOPTRACE_JAVA_PACKAGE}.PartitionDriftCorrectionRois"
    num_del = H.num_bead_rois_for_drift_correction
    num_acc = H.num_bead_rois_for_drift_correction_accuracy
    output_folder = output_folder or H.bead_rois_path
    if isinstance(output_folder, ExtantFolder):
        output_folder = output_folder.path
    cmd_parts = [
        "java", 
        "-cp",
        LOOPTRACE_JAR_PATH,
        prog_path, 
        "--beadRoisRoot",
        H.bead_rois_path, 
        "--numShifting", 
        num_del,
        "--numAccuracy",
        num_acc,
        "--outputFolder",
        output_folder,
    ]
    print(f"Running bead ROI partitioning: {' '.join(map(str, cmd_parts))}")
    subprocess.check_call(cmd_parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver for computing all fiducial bead ROIs for a particular imaging experiment")
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    parser.add_argument("-O", "--output-folder", type=Path, help="Path to folder in which to place output")
    