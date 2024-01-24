from math import floor
from pathlib import Path
from typing import *
import numpy as np
import pandas as pd


Interval = Tuple[int, int]
Box = Tuple[Interval, Interval, Interval]
Point = Tuple[float, float, float]


def get_bbox(r: Mapping[str, Any]) -> Box:
    z_box = (r["z_min"], r["z_max"])
    y_box = (r["y_min"], r["y_max"])
    x_box = (r["x_min"], r["x_max"])
    return z_box, y_box, x_box


down_to_int = lambda x: int(floor(x))


def extract_subimage(img: np.ndarray, sides: Box) -> np.ndarray:
    z, y, x = sides
    return img[slice(*z), slice(*y), slice(*x)]


def extract_for_roi(img: np.ndarray, df: pd.DataFrame, roi_id: int) -> np.ndarray:
    row = row_to_map(df[df.roi_id == roi_id])
    (z1, z2), (y1, y2), (x1, x2) = get_bbox(row)
    bbox = ((down_to_int(z1), down_to_int(z2)), (down_to_int(y1), down_to_int(y2)), (down_to_int(x1), down_to_int(x2)))
    return extract_subimage(img, sides=bbox)


def get_center_from_regions(df, roi_id) -> Point:
    row = row_to_map(df[df.index == roi_id])
    return row["zc"], row["yc"], row["xc"]


def get_center_from_extract(df, roi_id) -> Point:
    row = row_to_map(df[df.roi_id == roi_id])
    (z1, z2), (y1, y2), (x1, x2) = get_bbox(row)
    return 0.5 * (z1 + z2), 0.5 * (y1 + y2), 0.5 * (x1 + x2)


def row_to_map(r: pd.DataFrame) -> Mapping[str, Any]:
    assert r.shape[0] == 1, f"Not exactly 1 row! {r.shape[0]}"
    return {k: list(m.values())[0] for k, m in r.to_dict().items()}


def get_npy_file_name(posname: str, roi_id: int, ref_frame: int) -> str:
    return f"{posname}_{roi_id}_{ref_frame}.npy"





# rois_of_interest = [10345, 10349, 10350, 10351]
# time_of_interest = 57
# pos_of_interest = "P0013.zarr"

# experiment_root = Path(...)
# analysis_folder = experiment_root / ...
# diagnostics_folder = experiment_root / ...
# old_spot_images_folder = experiment_root / ... / ...
# rois = pd.read_csv(analysis_folder / ..., index_col=0)
# rois = rois[(rois.position == pos_of_interest) & (rois.frame == time_of_interest)]
# dcrois = pd.read_csv(analysis_folder / ..., index_col=0)
# dcrois = dcrois[(dcrois.position == pos_of_interest) & (dcrois.frame == time_of_interest) & (dcrois.ref_frame == time_of_interest)]

# all(get_center_from_regions(rois, i) == get_center_from_extract(dcrois, i) for i in rois_of_interest)

# bigimg = np.load(experiment_root / ...)
# imgs = {i: extract_for_roi(img=bigimg, df=dcrois, roi_id=i) for i in rois_of_interest}

# new_spot_images_folder = diagnostics_folder / ...

# def save_npy_file(arr: np.ndarray, roi_id: int) -> Path:
#     fp = new_spot_images_folder / get_npy_file_name(posname=pos_of_interest, roi_id=roi_id, ref_frame=time_of_interest)
#     np.save(fp, arr)
#     return fp

# new_spot_files = {i: save_npy_file(img, roi_id=i) for i, img in imgs.items()}
