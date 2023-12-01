"""Helper functions for analysing small spot images for tracing"""

import argparse
import itertools
from math import floor, ceil
from pathlib import Path
import re
from typing import *

import joblib
import numpy as np
import pandas as pd
import tqdm
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




###########################################################################################################
# Pinning down the cause of the all-0s-spot-image phenomenon (small ROIs used for tracing)
###########################################################################################################
def get_bounding_box_shape(roi: Union[Dict[str, NumberLike], pd.Series]) -> Tuple[int, int, int]:
    lo = lambda x: int(floor(x))
    hi = lambda x: int(ceil(x))
    return tuple(hi(roi[dim + "_max"]) - lo(roi[dim + "_min"]) for dim in ("z", "y", "x"))


def regional_shape_matches_initial_shape(one_roi_subframe: pd.DataFrame) -> bool:
    # Passed frame should be from DC ROIs table (*_dc_rois.csv), and should correspond to all rows from exactly 1 ROI.
    assert len(one_roi_subframe.roi_id.unique()) == 1, f"Subframe should have single ROI ID, got {len(one_roi_subframe.roi_id.unique())}"
    initial = get_bounding_box_shape(one_roi_subframe[one_roi_subframe.frame == 0])
    regional = get_bounding_box_shape(one_roi_subframe[one_roi_subframe.frame == one_roi_subframe.ref_frame])
    return initial == regional


def build_boolean_zeros_and_shape_agreement_table__regional_frames_only(T: Tracer, dc_rois: pd.DataFrame) -> pd.DataFrame:
    unique__roi_ref = dc_rois.drop_duplicates(subset=["roi_id", "ref_frame"])
    def get_one_row(r: pd.Series) -> Dict[str, NumberLike]:
        i = r["roi_id"]
        t = r["ref_frame"]
        shape_matches = regional_shape_matches_initial_shape(dc_rois[dc_rois.roi_id == i])
        is_all_zeros = T.images[i][t].max() == 0
        return {"pos_index": r["pos_index"], "roi_id": i, "ref_frame": t, "shape_match": shape_matches, "all_zeros": is_all_zeros}
    return pd.DataFrame(joblib.Parallel(n_jobs=-1, prefer='threads')(joblib.delayed(get_one_row)(row) for _, row in unique__roi_ref.iterrows()))    


def build_boolean_zeros_and_shape_agreement_table__regional_frames_only(T: Tracer, dc_rois: pd.DataFrame) -> pd.DataFrame:
    unique__roi_ref = dc_rois.drop_duplicates(subset=["roi_id", "ref_frame"])
    def get_one_roi_rows(ref_roi: pd.Series) -> Dict[str, NumberLike]:
        pid = ref_roi["pos_index"]
        rid = ref_roi["roi_id"]
        sub = dc_rois[dc_rois.roi_id == rid]
        init_shape = get_bounding_box_shape(sub[sub.frame == 0])
        return [{
            "pos_index": pid, 
            "roi_id": rid, 
            "ref_frame": ref_roi["frame"], 
            "shape_match": get_bounding_box_shape(roi) == init_shape, 
            "all_zeros": T.images[rid][roi["frame"]].max() == 0
            } 
            for _, roi in sub.iterrows()
            ]
    return pd.DataFrame(itertools.chain(
        *joblib.Parallel(n_jobs=-1, prefer='threads')(joblib.delayed(get_one_roi_rows)(row) for _, row in tqdm.tqdm(unique__roi_ref.iterrows()))
        ))


def shape_match_all_zeros_xor(df: pd.DataFrame) -> pd.DataFrame:
    # (a AND NOT b) OR (NOT a AND b)
    return ((df.shape_match & ~df.all_zeros) | (~df.shape_match & df.all_zeros)).all()
###########################################################################################################
# Pinning down the cause of the all-0s-spot-image phenomenon (small ROIs used for tracing)
###########################################################################################################


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
