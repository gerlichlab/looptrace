"""Wrappers around functions and types, to tailor them to the use cases of this package"""

__author__ = "Vince Reuter"

__all__ = ["phase_cross_correlation"]

from typing import *
import numpy as np
from skimage.registration import phase_cross_correlation as phase_xcor


def phase_cross_correlation(ref_img: np.ndarray, mov_img: np.ndarray, upsample_factor: int = 1) -> Tuple[np.ndarray, float, float]:
    """
    Run skimage.registration.phase_cross_correlation from scikit-image, fixing the parameter which controls return value structure.

    See: https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.phase_cross_correlation

    Parameters
    ----------
    ref_img : np.ndarray
        The image to which the other should be shifted / adjusted to be aligned
    mov_img : np.ndarray
        The image that's shifted relative to reference
    upsample_factor : int, optional
        Factor for upsampling, defaulting to 1 (no upsampling)

    Returns
    -------
    tuple of np.ndarray, float, float
        A trio of values in which the first is the vector representing the shift needed to align the moving image to the 
        reference image, the second is the error, and the third is the global phase difference.
    """
    return phase_xcor(reference_image=ref_img, moving_image=mov_img, upsample_factor=upsample_factor, return_error='always')
