"""Run the labeling and filtering of the trace supports."""

import argparse
import itertools
import subprocess
from typing import *

from gertils import ExtantFile, ExtantFolder

from looptrace import *
from looptrace.ImageHandler import handler_from_cli

__author__ = "Vince Reuter"


def workflow(config_file: ExtantFile, images_folder: ExtantFolder) -> None:
    H = handler_from_cli(config_file=config_file, images_folder=images_folder)
    min_spot_sep = H.minimum_spot_separation
    region_groups = H.config.get("regional_spots_grouping", "NONE")
    if not isinstance(region_groups, str):
        region_groups = ",".join(",".join([f"{r}={i}" for r in rs]) for i, rs in enumerate(region_groups))
    if min_spot_sep <= 0:
        print(f"No spot filtration on proximity to be done, as minimum separation = {min_spot_sep}")
    else:
        prog_path = f"{LOOPTRACE_JAVA_PACKAGE}.LabelAndFilterRois"
        cmd_parts = [
            "java", 
            "-cp",
            str(LOOPTRACE_JAR_PATH.path),
            prog_path, 
            "--spotsFile",
            str(H.raw_spots_file),
            "--driftFile", 
            str(H.drift_correction_file__fine),
            "--probeGroups",
            region_groups,
            "--spotSeparationThresholdValue", 
            str(H.minimum_spot_separation),
            "--spotSeparationThresholdType",
            "EachAxisAND",
            "--unfilteredOutputFile",
            str(H.proximity_labeled_spots_file_path),
            "--filteredOutputFile",
            str(H.proximity_filtered_spots_file_path),
            "--handleExtantOutput",
            "OVERWRITE" # TODO: parameterise this, see: https://github.com/gerlichlab/looptrace/issues/142
        ]
        print(f"Running spot filtering on proximity: {' '.join(cmd_parts)}")
        subprocess.check_call(cmd_parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver for filtering out too-proximal regional spots")
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    args = parser.parse_args()
    workflow(config_file=args.config_path, images_folder=args.image_path)
