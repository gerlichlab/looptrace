import logging
import os
from pathlib import Path
from typing import *

import napari
import numpy as np
import pandas as pd
from skimage.io import imsave

from looptrace.numeric_types import NumberLike

__author__ = "Vince Reuter"
__credits__ = ["Vince Reuter"]

__all__ = [
    "SIGNAL_TO_QUIT", 
    "add_points_to_viewer", 
    "add_screenshot", 
    "extract_roi_centers", 
    "prompt_continue_napari",
    "save_screenshot",
    "shutdown_napari",
    ]

logger = logging.getLogger()


SIGNAL_TO_QUIT = "q"


def add_points_to_viewer(
    viewer: napari.Viewer,
    points: np.ndarray, 
    *,
    size: Union[NumberLike, list, np.array], 
    symbol: str = "square", # Usually we want a bounding box.
    edge_width: NumberLike = 1, 
    edge_width_is_relative: bool = False,
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
    symbol : str
        The name of the shape of the points to draw

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
        symbol=symbol,
        edge_width=edge_width,
        edge_width_is_relative=edge_width_is_relative,
        n_dimensional=n_dimensional,
        **kwargs,
        )


def add_screenshot(viewer: napari.Viewer, *, flash: bool = False, **kwargs) -> np.ndarray:
    """Take a screenshot with the given viewer, and add and return the resulting image."""
    screenshot = viewer.screenshot(flash=flash, **kwargs)
    viewer.add_image(screenshot)
    return screenshot


def extract_roi_centers(rois: pd.DataFrame) -> np.ndarray:
    return rois[["zc", "yc", "xc"]].to_numpy()


def prompt_continue_napari() -> str:
    return input(f"Close window when done, then press enter to continue to the next image or {SIGNAL_TO_QUIT} to quit.")


def save_screenshot(viewer: napari.Viewer, outfile: Union[str, Path], *, flash: bool = False, **kwargs):
    screenshot = add_screenshot(viewer, flash=flash, **kwargs)
    os.makedirs(outfile.parent, exist_ok=True)
    return imsave(outfile, screenshot)


def shutdown_napari() -> None:
    """Close remaining napari windows to prepare for clean program exit."""
    logger.warn("Closing any remaining napari windows...")
    napari.Viewer.close_all()
