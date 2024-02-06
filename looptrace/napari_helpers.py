import logging
from typing import *

import numpy as np
import pandas as pd
import napari

from looptrace.numeric_types import NumberLike

__author__ = "Vince Reuter"
__credits__ = ["Vince Reuter"]

__all__ = [
    "SIGNAL_TO_QUIT", 
    "add_points_to_viewer", 
    "add_screenshot", 
    "extract_roi_centers", 
    "shutdown_napari",
    ]

logger = logging.getLogger()


SIGNAL_TO_QUIT = "q"


def add_points_to_viewer(
    viewer: napari.Viewer,
    points: np.ndarray, 
    *,
    size: Union[NumberLike, list, np.array], 
    edge_width: NumberLike = 1, 
    edge_width_is_relative: bool = False,
    symbol: str = "square",
    edge_color: str = "red", 
    face_color: str = "transparent", 
    n_dimensional: bool = False,
    **kwargs,
    ) -> napari.layers.Points:
    """
    Parameters
    ----------
    viewer : napari.Viewer
        The image viewer to which to add the points layer
    points : np.ndarray
        The ROI centers or other points to be added to the image viewer
    size : looptrace.numeric_types.NumberLike or np.array
        Either a single value to make all points the same size, or an array of shape compatible with points, 
        e.g. a 1D array of length equal to the number of points.

    Returns
    -------
    napari.layers.Points
        The layer of points added to the image viewer

    Other Parameters
    ----------------
    See the docs for napari.Viewer.add_points
    """
    return viewer.add_points(
        points, 
        size=size,
        edge_width=edge_width,
        edge_width_is_relative=edge_width_is_relative,
        symbol=symbol,
        edge_color=edge_color,
        face_color=face_color,
        n_dimensional=n_dimensional,
        **kwargs,
        )


def add_screenshot(viewer: napari.Viewer) -> np.ndarray:
    """Take a screenshot with the given viewer, and add and return the resulting image."""
    screenshot = viewer.screenshot()
    viewer.add_image(screenshot)
    return screenshot


def extract_roi_centers(rois: pd.DataFrame) -> np.ndarray:
    return rois[["zc", "yc", "xc"]].to_numpy()


def shutdown_napari() -> None:
    """Close remaining napari windows to prepare for clean program exit."""
    logger.warn("Closing any remaining napari windows...")
    napari.Viewer.close_all()
