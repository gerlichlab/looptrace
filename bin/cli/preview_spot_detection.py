"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import argparse
import logging
from typing import *

import napari
import numpy as np
import pandas as pd

from gertils import ExtantFile

from looptrace.ImageHandler import ImageHandler
from looptrace.SpotPicker import SpotPicker, compute_downsampled_image
from looptrace.image_processing_functions import subtract_crosstalk
from looptrace.napari_helpers import add_points_to_viewer, extract_roi_centers

__author__ = "Kai Sandvold Beckwith"
__credits__ = ["Kai Sandvold Beckwith", "Vince Reuter"]

logger = logging.getLogger(__name__)


def _roi_to_napari_points(roi_table: pd.DataFrame, position: str) -> np.ndarray:
    """Convert roi from looptrace roi table to points to see in napari.

    Parameters
    ----------
    roi_table : pd.DataFrame
        ROI data found in looptrace pipeline
    position : str
        Positional (FOV) identifier

    Returns
    -------
    np.ndarray, dict
        Numpy array of shape N x 4, with the 4 volumns (frame, z, y, x), and a dict of the roi_ids
    """
    rois_at_pos = roi_table[roi_table["position"] == position]
    roi_shapes = extract_roi_centers(roi_table)
    roi_shapes = [[r["zc"], r["yc"], r["xc"]] for _, r in rois_at_pos.iterrows()]
    return np.array(roi_shapes), {"roi_id": rois_at_pos["roi_id_pos"].values}


def napari_view(
    images: Union[list[np.ndarray], dict[str, np.ndarray]], 
    points: np.ndarray, 
    *, 
    axes: str, 
    downscale: int, 
    roi_size: int, 
    roi_symbol: str = "square",
    ) -> napari.layers.Points:
    """View images as layers in napari, adding points on top.

    Parameters
    ----------
    images : list or dict
        Images in which spots are to be detected, either a simple array or a mapping from name to array
    points : np.ndarray
        Points to add to underling images, representing detected spots
    axes : str
        Names of the dimensions of the images and points being viewed
    downscale : int
        How to subsample (step size) the images and points to view
    roi_size : int
        Desired side length of square or diameter of circle in full (not downscaled) space

    Returns
    -------
    napari.layers.Points
        The layers of detected spot ROIs
    """
    import napari

    # Ensure there's actual image data to view.
    if not images:
        raise ValueError(f"Non-null, non-empty images collection must be passed to view with napari")
    # Unpack images and names.
    if isinstance(images, dict):
        names, images = zip(*images.items())
    elif isinstance(images, list):
        names = None
    else:
        raise TypeError(f"Collection of images to view should be list or dict, not {type(images).__name__}!")
    # Check that each image is a numpy array of same dimensionality, and that it's 2D or 3D.
    if not all(isinstance(img, np.ndarray) for img in images):
        raise TypeError(f"Each image to view should be a numpy array; got: {', '.join(type(img).__name__ for img in images)}")
    num_dim = len(axes)
    shape = images[0].shape
    if not len(shape) == num_dim:
        raise ValueError(f"Each image must have {num_dim} dimensions to conform with axes {axes}; got {len(shape)} for first image: {shape}")
    if not all(img.shape == shape for img in images):
        raise ValueError(
            f"Each images to view must have shape {shape}; got {', '.join(img.shape for img in images)}"
            )
    
    # Do the image viewing and points layer addition.
    images = np.stack(images)
    viewer = napari.view_image(images, channel_axis=0, name=names)
    point_layer = add_points_to_viewer(
        viewer=viewer, 
        points=points/downscale, 
        size=roi_size/downscale, 
        symbol=roi_symbol, 
        edge_color="red", 
        face_color="transparent", 
        )
    sel_dim = list(points[0, :] / downscale)
    for dim in range(len(sel_dim)):
        viewer.dims.set_current_step(dim, sel_dim[dim])
    napari.run()
    return point_layer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise preview spot detection in given position.")
    parser.add_argument("rounds_config", type=ExtantFile.from_string, help="Imaging rounds config file path")
    parser.add_argument("params_config", type=ExtantFile.from_string, help="Looptrace parameters config file path")
    parser.add_argument("images_folder", help="Path to folder with images to read.")
    parser.add_argument('--position', type=int, help='(Optional): Index of position to view.', default=0)
    args = parser.parse_args()
        
    logger.info(f"Building image handler: {args.images_folder}")
    H = ImageHandler(
        rounds_config=args.rounds_config,
        params_config=args.params_config, 
        images_folder=args.images_folder,
        )
    S = SpotPicker(H)
    params = S.detection_parameters
    for (i, frame), ch in S.iter_frames_and_channels():
        logger.info(f"Previewing spot detection in position {args.position}, frame {frame} with threshold {S.spot_threshold[i]}...")
        img = compute_downsampled_image(S.images[args.position], frame=frame, channel=ch, downsampling=S.downsampling)
        if params.subtract_beads:
            bead_img = compute_downsampled_image(S.images[args.position], frame=frame, channel=params.crosstalk_channel, downsampling=S.downsampling)
            img, orig = subtract_crosstalk(source=img, bleed=bead_img, threshold=0)
        spot_detection_result = params.detection_function(img, threshold=S.spot_threshold[i])
        spot_props = spot_detection_result.table
        filt_img = spot_detection_result.image
        spot_props["position"] = S.pos_list[args.position]
        spot_props = spot_props.reset_index().rename(columns={"index": "roi_id_pos"})
        spot_props = params.try_centering_spot_box_coordinates(spots_table=spot_props)
        images_to_view = {"DoG": filt_img, "Subtracted": img, "Original": orig} if params.subtract_beads else {"DoG": filt_img, "Original": img}
        roi_points, _ = _roi_to_napari_points(spot_props, position=S.pos_list[args.position])
        if H.roi_image_size.y != H.roi_image_size.x:
            roi_size = (H.roi_image_size.y + H.roi_image_size.x) / 2
            logger.warn(f"ROI size differs in y ({H.roi_image_size.y}) and x ({H.roi_image_size.x}). Will use average: {roi_size}")
        else:
            roi_size = H.roi_image_size.y
        roi_size = H.roi_image_size.y / S.downsampling
        logger.debug(f"ROI size after downsampling: {roi_size}")
        napari_view(images_to_view, roi_points, axes="ZYX", downscale=1, roi_size=roi_size)
