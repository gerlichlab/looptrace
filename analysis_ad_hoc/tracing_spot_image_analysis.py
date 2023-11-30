"""Helper functions for analysing small spot images for tracing"""

import argparse
from pathlib import Path
import re
from typing import *

import numpy as np
import pandas as pd
from gertils import ExtantFile, ExtantFolder
from looptrace.ImageHandler import ImageHandler
from looptrace.Tracer import Tracer
from looptrace.numeric_types import NumberLike


__author__ = "Vince Reuter"

POSITION_NAME_PATTERN = re.compile(r"\bP[0]*(\d+(?:)?).zarr\b") # Require P + any 0s and .zarr to flank positive int.


def get_roi_id_and_region_frame(spot_file_name: str) -> Dict[str, int]:
    try:
        prefix, roi_id_text, reg_frame_text = spot_file_name.split("_")
    except ValueError:
        print(f"ERROR: could not unpack filds for prefix, ROI ID, and regional barcode from filename: {spot_file_name}")
    pos_name_match = re.search(POSITION_NAME_PATTERN, prefix)
    if pos_name_match is None:
        raise ValueError(f"Could not parse position name from prefix '{prefix}' from filename: {spot_file_name}")
    pos = int(pos_name_match.group(1)) - 1 # 1-based to 0-based
    roi = int(roi_id_text)
    reg = int(reg_frame_text)
    return {"pos_index": pos, "roi_id": roi, "ref_frame": reg}


def get_stats_single_timepoint(single_roi_time_stack: np.ndarray) -> Dict[str, NumberLike]:
    if len(single_roi_time_stack.shape) != 3:
        raise ValueError(f"Expected a 3D array but got {single_roi_time_stack.shape} for shape!")
    return {
        "max": single_roi_time_stack.max(),
        "mean": single_roi_time_stack.mean(), 
        "median": np.median(single_roi_time_stack),
        }


def get_stats_single_roi(single_roi_stack: np.ndarray) -> Iterable[Dict[str, NumberLike]]:
    if len(single_roi_stack.shape) != 4:
        raise ValueError(f"Expected a 4D array but got {single_roi_stack.shape} for shape!")
    for t in range(single_roi_stack.shape[0]):
        yield {"frame": t} | get_stats_single_timepoint(single_roi_stack[t])


def get_stats_all_rois(T: Tracer) -> Iterable[Dict[str, NumberLike]]:
    for fn, img in zip(T.images.files, T.images):
        meta = get_roi_id_and_region_frame(fn)
        data = get_stats_single_roi(img)
        for stat in data:
            yield meta | stat


def get_spot_image_stats_for_experiment(conf: Union[str, Path, ExtantFile], imgs: Union[str, Path, ExtantFolder]) -> Iterable[Dict[str, NumberLike]]:
    print("Building Tracer instance...")
    T = Tracer(ImageHandler(conf, imgs))
    print("Computing stats table...")
    return pd.DataFrame(get_stats_all_rois(T))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Produced aggregation statistics about the spot image files used for tracing.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument("looptrace_config_file", type=ExtantFile.from_string, help="Path to looptrace config file for an experiment")
    parser.add_argument("looptrace_images_folder", type=ExtantFolder.from_string, help="Path to images folder used for looptrace experiment")
    parser.add_argument("-O", "--outfile", required=True, help="Path to output file to write")
    
    args = parser.parse_args()

    stats_table = get_spot_image_stats_for_experiment(conf=args.looptrace_config_file, imgs=args.looptrace_images_folder)
    print(f"Writing output file: {args.outfile}")
    stats_table.to_csv(args.outfile)
