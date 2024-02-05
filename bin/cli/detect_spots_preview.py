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

from looptrace.ImageHandler import ImageHandler
from looptrace.SpotPicker import SpotPicker
from looptrace.image_processing_functions import subtract_crosstalk

__author__ = "Kai Sandvold Beckwith"
__credits__ = ["Kai Sandvold Beckwith", "Vince Reuter"]

logger = logging.getLogger(__name__)


def roi_to_napari_points(roi_table, position, use_time: bool) -> np.ndarray:
    """Convert roi from looptrace roi table to points to see in napari.

    Parameters
    ----------
    roi_table : pd.DataFrame
        ROI data found in looptrace pipeline
    position : str
        Positional (FOV) identifier
    use_time : bool
        Whether or not to try to add a value (at the beginning of each point row) representing timepoint

    Returns
    -------
    np.ndarray, dict
        Numpy array of shape N x 4, with the 4 volumns (frame, z, y, x), and a dict of the roi_ids
    """

    rois_at_pos = roi_table[roi_table["position"] == position]
    finalise = (lambda row, sub: [row["frame"]] + sub) if use_time else (lambda _, sub: sub)
    roi_shapes = [finalise(r, [r["zc"], r["yc"], r["xc"]]) for _, r in rois_at_pos.iterrows()]
    return np.array(roi_shapes), {"roi_id": rois_at_pos["roi_id_pos"].values}


def napari_view(
    images: Union[list[np.ndarray], dict[str, np.ndarray]], 
    points: np.ndarray, 
    *, 
    axes: str, 
    downscale: int, 
    roi_size: int, 
    roi_symbol: str = "square",
    ) -> napari.layers.points.points.Points:
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
    napari.Points
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
    
    images = np.stack(images)
    print(f"DEBUG: Stacked images' shape: {images.shape}")
    viewer = napari.view_image(images, channel_axis=0, name=names)

    # DEBUG
    print(f"Points: {points}")
        
    point_layer = viewer.add_points(
        points / downscale, 
        size=roi_size / downscale,
        edge_width=1,
        edge_width_is_relative=False,
        symbol=roi_symbol,
        edge_color="red",
        face_color="transparent",
        n_dimensional=False,
        )
    sel_dim = list(points[0, :] / downscale)
    for dim in range(len(sel_dim)):
        viewer.dims.set_current_step(dim, sel_dim[dim])
    napari.run()
    return point_layer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preview spot detection in given position.')
    parser.add_argument("config_path", help="Config file path")
    parser.add_argument("image_path", help="Path to folder with images to read.")
    parser.add_argument('--position', type=int, help='(Optional): Index of position to view.', default=0)
    args = parser.parse_args()
        
    logger.info(f"Building image handler: {args.image_path}")
    H = ImageHandler(config_path=args.config_path, image_path=args.image_path)
    S = SpotPicker(H)
    params = S.detection_parameters
    for (i, frame), ch in S.iter_frames_and_channels():
        logger.info(f"Previewing spot detection in position {args.position}, frame {frame} with threshold {S.spot_threshold[i]}...")
        img = S.images[args.position][frame, ch, ::params.downsampling, ::params.downsampling, ::params.downsampling].compute()
        if params.subtract_beads:
            bead_img = S.images[args.position][frame, params.crosstalk_channel, ::params.downsampling, ::params.downsampling, ::params.downsampling].compute()
            img, orig = subtract_crosstalk(source=img, bleed=bead_img, threshold=0)
        spot_props, filt_img, _ = params.detection_function(img, S.spot_threshold[i])
        spot_props["position"] = S.pos_list[args.position]
        spot_props = spot_props.reset_index().rename(columns={"index": "roi_id_pos"})
        spot_props = params.try_centering_spot_box_coordinates(spots_table=spot_props)
        images_to_view = {"DoG": filt_img, "Subtracted": img, "Original": orig} if params.subtract_beads else {"DoG": filt_img, "Original": img}

        # DEBUG
        print(f"spot_props.shape: {spot_props.shape}")

        roi_points, _ = roi_to_napari_points(spot_props, position=S.pos_list[args.position], use_time=False)

        # DEBUG
        print(f"roi_points.shape: {roi_points.shape}")

        _, roi_sz_y, roi_sz_x = S.roi_image_size
        roi_size = (roi_sz_y + roi_sz_x) / 2
        if roi_sz_y != roi_sz_x:
            logger.warn(f"ROI size differs in y ({roi_sz_y}) and x ({roi_sz_x}). Will use average: {roi_size}")
        roi_size = roi_sz_y / params.downsampling
        logger.debug(f"ROI size after downsampling: {roi_size}")
        napari_view(images_to_view, roi_points, axes="ZYX", downscale=1, roi_size=roi_size)
