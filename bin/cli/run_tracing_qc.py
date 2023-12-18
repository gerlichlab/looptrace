"""Run the labeling and filtering of the trace supports."""

import argparse
import subprocess
from typing import *

from gertils import ExtantFile, ExtantFolder

from looptrace import *
from looptrace.ImageHandler import handler_from_cli
from looptrace.Tracer import Tracer

__author__ = "Vince Reuter"


def workflow(config_file: ExtantFile, images_folder: ExtantFolder) -> None:
    H = handler_from_cli(config_file=config_file, images_folder=images_folder)
    T = Tracer(H)
    prog_path = f"{LOOPTRACE_JAVA_PACKAGE}.LabelAndFilterTracesQC"
    cmd_parts = [
        "java", 
        "-cp",
        str(LOOPTRACE_JAR_PATH),
        prog_path, 
        "--tracesFile",
        str(T.traces_path_enriched),
        "--maxDistanceToRegionCenter", 
        str(H.config[MAX_DISTANCE_SPOT_FROM_REGION_NAME]),
        "--minSNR",
        str(H.config[SIGNAL_NOISE_RATIO_NAME]),
        "--maxSigmaXY",
        str(H.config[SIGMA_XY_MAX_NAME]),
        "--maxSigmaZ",
        str(H.config[SIGMA_Z_MAX_NAME]),
    ]

    exclusions_key = "illegal_frames_for_trace_support"
    probes_to_exclude = H.config.get(exclusions_key, [])
    if probes_to_exclude:
        if not isinstance(probes_to_exclude, list):
            raise TypeError(f"Probes to exclude ('{exclusions_key}' in config) should be list, got {type(probes_to_exclude).__name__}")
        cmd_parts.extend(["--exclusions", ','.join(probes_to_exclude)]) # format required for parsing by scopt
    else:
        print("WARNING! No probes to exclude from trace support were provided!")
    
    print(f"Running QC filtering of tracing supports: {' '.join(cmd_parts)}")
    subprocess.check_call(cmd_parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver for filtering tracing supports")
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    args = parser.parse_args()
    workflow(config_file=args.config_path, images_folder=args.image_path, )
