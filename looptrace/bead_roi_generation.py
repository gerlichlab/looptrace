"""Generation of ROIs from signal from fiducial beads

Fiducial beads are small points of flurescence which should be positioned identically 
in each of the images within a field of view, throughout the time coarse of the imaging 
experiment. As such, they can be used to align images to correct for drift.

These are generated by adding a small quantity of diluted solution that will produce 
small points of fluorescence at an expected wavelength when excited at a particular 
intensity. In a particular imaging channel, we then detect these points of light 
and generate regions of interest (ROIs) corresponding to them.
"""

__author__ = "Vince Reuter"

import dataclasses
from enum import Enum
from joblib import Parallel, delayed
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from skimage.measure import regionprops_table
import tqdm

from gertils import ExtantFolder

from looptrace import (
    X_CENTER_COLNAME, 
    Y_CENTER_COLNAME, 
    Z_CENTER_COLNAME,
    ArrayDimensionalityError,
)
from looptrace.ImageHandler import bead_rois_filename
from looptrace.filepaths import simplify_path
from looptrace.geometry import Point3D
from looptrace.image_processing_functions import (
    CENTROID_KEY, 
    get_centroid_column_name_remapping,
)
from looptrace.numeric_types import NumberLike

PathLike = Union[str, Path]


AREA_COLUMN_NAME = "area"
FAIL_CODE_COLUMN_NAME = "fail_code"
INDEX_COLUMN_NAME = "beadIndex"
INTENSITY_COLUMN_NAME = "max_intensity"


def extract_single_bead(
    point: Point3D, 
    img: np.ndarray, 
    bead_roi_px: int = 16, 
    drift_coarse: Optional[Point3D] = None,
) -> np.ndarray:
    """
    Extract a cropped region of a single fiducial in an image, optionally including a pre-calucalated coarse drift to shift the cropped region.

    Parameters
    ----------
    point : Point3D
        Coordinates representing the center of a detected bead
    img : np.ndarray
        An array of values representing an image in which a fiducial bead is detected
    bead_roi_px : int
        The number of pixels for the side length of the bead ROI
    drift_coarse : None or Point3D
        The coarse-grained drift correction already computed

    Returns
    -------
    np.ndarray
        A numpy array representing the subspace of the given image corresponding to the bead ROI
    """
    # TODO: assert that both point and the drift vector are 3D.
    # NB: here we subtract the coarse drift from the point. This is because of discussion in #194.
    #     Essentially, per scikit-learn and phase_cross_correlation, offset = ref_img - mov_img
    #     Therefore, mov_img = ref_img - offset, so to get coordinates in the "moving" space, 
    #     we subtract the shift from the reference point.
    # See: https://github.com/gerlichlab/looptrace/issues/194
    roi_px = bead_roi_px // 2
    # The image from which to pull the voxel is in (z, y, x) sequence.
    flatten_point = lambda p: [p.z, p.y, p.x]
    point = flatten_point(point)
    coords: list[int | float] = [
        int(round(c)) 
        for c in (
            point if drift_coarse is None else \
            [x - dx for x, dx in zip(point, flatten_point(drift_coarse), strict=True)]
        )
    ]
    s = tuple([slice(p - roi_px, p + roi_px) for p in coords])
    bead = img[s]
    side_length = 2 * roi_px
    output_shape = (side_length, side_length, side_length)
    # TODO: consider, for provenance, logging a message here, that the bead shape was not as expected, and all-0s is used.
    return np.zeros(output_shape) if bead.shape != output_shape else bead


def validate_array_dimension(arr: np.ndarray, expected: int) -> None:
    obs = len(arr.shape)
    if obs != expected:
        raise ArrayDimensionalityError(f"Expected {expected} dimensions, but got {obs}")


def get_dimension(obj) -> Optional[int]:
    try:
        shape = obj.shape
    except AttributeError:
        return None
    return len(shape)


def iterate_over_pos_time_images(image_array: List[np.ndarray], channel: Optional[int] = None) -> Iterable[Tuple[int, np.ndarray]]:
    if isinstance(image_array, list) and all(get_dimension(arr) == 5 for arr in image_array): # includes dimension for channel (t, c, z, y, x)
        if channel is None:
            raise ValueError("Must specify beads channel when iterating over 5-tensors.")
        for p, pos_imgs in enumerate(image_array):
            for t in range(pos_imgs.shape[0]):
                yield (p, t), pos_imgs[t, channel, :, :, :]
    elif isinstance(image_array, list) and all(get_dimension(arr) == 4 for arr in image_array):
            for p, pos_imgs in enumerate(image_array):
                for t in range(pos_imgs.shape[0]):
                    yield (p, t), pos_imgs[t, :, :, :]
    else:
        raise TypeError(f"Illegal image array for positional iteration: {type(image_array).__name__}")


def generate_all_bead_rois_from_getter(
    get_3d_stack: Callable[[int, int],  np.ndarray], 
    iter_fov: Iterable[int], 
    iter_timepoint: Iterable[int], 
    output_folder: Union[str, Path, ExtantFolder], 
    params: "BeadRoiParameters", 
    **joblib_kwargs,
    ) -> List[Tuple[Path, pd.DataFrame]]:
    
    output_folder: Path = simplify_path(output_folder)

    def get_outfile(fov_idx: int, timepoint_idx: int) -> Path:
        fn: str = bead_rois_filename(fov_idx=fov_idx, timepoint=timepoint_idx, purpose=None)
        return output_folder / fn
    
    def proc1(img: np.ndarray, outfile: Path) -> tuple[Path, pd.DataFrame]:
        rois = params.compute_labeled_regions(img=img)
        print(f"Writing ROIs: {outfile}")
        rois.to_csv(outfile, index_label=INDEX_COLUMN_NAME) # This call generates each bead_rois__<FOV>_<TIME>.csv file.
        return outfile, rois
    
    return Parallel(**joblib_kwargs)(
        delayed(proc1)(img=get_3d_stack(fov_idx, timepoint), outfile=get_outfile(fov_idx=fov_idx, timepoint_idx=timepoint)) 
        for fov_idx in tqdm.tqdm(iter_fov) for timepoint in tqdm.tqdm(iter_timepoint)
        )


def generate_bead_rois(
    t_img: np.ndarray, 
    threshold: NumberLike, 
    min_bead_int: NumberLike, 
    bead_roi_px: int = 16, 
    n_points: int = 200, 
    max_size: NumberLike = 500, 
    max_bead_int: Optional[NumberLike] = None, 
    **kwargs,
) -> list[Point3D]:
    '''Function for finding positions of beads in an image based on manually set thresholds in config file.

    Parameters
    ----------
    t_img (3D ndarray): The image from which to generate the bead ROIs
    threshold (float): Minimum value needed to count a pixel as potentially part of a bead
    min_bead_int (float): Minimum value (average, or maximum?) in segmented region needed for it to be counted as a bead
    n_points (int): How many bead ROIs to return
    max_size (int): The maximum area of a bead region
    max_bead_int (int): Optionally, the maximum value to tolerate and still count a segmented region as a bead

    Returns:
    t_img_maxima: a collection of 3D bead coordinates in t_img
    '''
    params = BeadRoiParameters(
        min_intensity_for_segmentation=threshold, 
        min_intensity_for_detection=min_bead_int, 
        roi_pixels=bead_roi_px, 
        max_region_size=max_size, 
        max_intensity_for_detection=max_bead_int
        )
    return params.generate_image_rois(img=t_img, num_points=n_points, **kwargs)


@dataclasses.dataclass
class BeadRoiParameters:
    min_intensity_for_segmentation: NumberLike
    min_intensity_for_detection: NumberLike
    roi_pixels: int
    max_region_size: NumberLike
    max_intensity_for_detection: Optional[NumberLike] = None

    def generate_image_rois(
        self, 
        img: np.ndarray, 
        num_points: int, 
        filtered_filepath: Optional[PathLike] = None, 
        unfiltered_filepath: Optional[PathLike] = None,
    ) -> list[Point3D]:
        """
        Parameters
        ----------
        img : np.ndarray
            3D array in which each value is a pixel intensity
        num_points : int
            How many bead ROIs to return
        filtered_filepath : str or Path, optional
            Path to which to write the ROIs chosen / sampled; 
            if unspecified, don't write anything
        unfiltered_filepath : str or Path, optional
            Path to which to write the ROIs before sampling, but with QC label based on detection criteria; 
            if unspecified, don't write anything

        Returns
        -------
        Iterable of Point3D
            3D bead coordinates in given image
        """
        
        # Add the fail_code field, based on reason(s) to exclude a detected bead from selection.
        img_maxima = self.compute_labeled_regions(img=img)

        fail_codes = img_maxima[FAIL_CODE_COLUMN_NAME]

        if unfiltered_filepath:
            print(f"Writing unfiltered bead ROIs: {unfiltered_filepath}")
            unfiltered_filepath.parent.mkdir(parents=False, exist_ok=True)
            img_maxima.to_csv(unfiltered_filepath, index_label=INDEX_COLUMN_NAME)

        print("Filtering bead ROIs")
        num_unfiltered = len(img_maxima)
        img_maxima = img_maxima[(fail_codes == "") | fail_codes.isna()]
        num_filtered = len(img_maxima)
        print(f"Bead ROIs remaining: {num_filtered}/{num_unfiltered}")

        if num_filtered == 0:
            print("FAIL CODES...")
            print(fail_codes)

        # Sample the ROIs, or retain all of them, depending on the config setting for number of points, and 
        # the number of regions which satisfy the filtration criteria.
        if num_points == -1:
            print(f"Using all bead ROIs based on setting for number of points")
        elif num_filtered <= num_points:
            print(f"Using all bead ROIs based on number remaining: {num_filtered} <= {num_points}")
        else:
            print(f"Sampling bead ROIs: {num_points}/{num_filtered}")
            img_maxima = img_maxima.sample(n=num_points, random_state=1)
        
        if filtered_filepath:
            print(f"Writing sampled bead ROIs: {filtered_filepath}")
            img_maxima.to_csv(filtered_filepath, index_label=INDEX_COLUMN_NAME)

        return [
            Point3D(
                z=roi[Z_CENTER_COLNAME], 
                y=roi[Y_CENTER_COLNAME], 
                x=roi[X_CENTER_COLNAME],
            )
            for _, roi in img_maxima.iterrows()
        ]

    def compute_labeled_regions(self, img: np.ndarray) -> pd.DataFrame:
        """Find contiguous regions (according to instance settings) within given image, and assign fail code(s)."""
        if not isinstance(img, np.ndarray):
            raise TypeError(f"Image must be 3D array; got {type(img).__name__}")
        if 3 != len(img.shape):
            raise TypeError(f"Image must be 3D array; got dimension of {len(img.shape)}")
        
        # Segment the image into contiguous regions of signal above the current threshold.
        img_maxima: pd.DataFrame = self._extract_regions(img)

        # Convert the scikit-image property names to more meaningful column names.
        colname_remapping: Mapping[str, str] = get_centroid_column_name_remapping(ndim=3)
        img_maxima = img_maxima.rename(columns=colname_remapping)
        
        # Apply failure code labels based on the regional filtration criteria.
        img_maxima[FAIL_CODE_COLUMN_NAME] = self._compute_discard_reasons(regions=img_maxima)

        return img_maxima

    def _extract_regions(self, img: np.ndarray) -> pd.DataFrame:
        # Segment the given image into regions of pixels in which the signal intensity exceeds the segmentation threshold.
        img_label, num_labels = self._segment_image(img)
        print("Number of unfiltered beads found: ", num_labels)
        # Extract the relevant data for each of the segmented regions.
        return pd.DataFrame(regionprops_table(img_label, img, properties=(CENTROID_KEY, INTENSITY_COLUMN_NAME, AREA_COLUMN_NAME)))

    def _compute_discard_reasons(self, regions: pd.DataFrame) -> pd.Series:
        # TODO: why divide-by-2 here?
        roi_px = self.roi_pixels // 2
        # TODO: record better the mapping from -0/-1/-2 to z/y/x.
        too_high = (lambda _: False) if self.max_intensity_for_detection is None \
            else (lambda row: row [INTENSITY_COLUMN_NAME] > self.max_intensity_for_detection)
        invalidation_label_pairs = [
            (lambda row: row[Z_CENTER_COLNAME] <= roi_px, self.BeadFailReason.OutOfBoundsZ), 
            (lambda row: row[Y_CENTER_COLNAME] <= roi_px, self.BeadFailReason.OutOfBoundsY), 
            (lambda row: row[X_CENTER_COLNAME] <= roi_px, self.BeadFailReason.OutOfBoundsX), 
            (lambda row: row[AREA_COLUMN_NAME] > self.max_region_size, self.BeadFailReason.TooBig), 
            (lambda row: row[INTENSITY_COLUMN_NAME] < self.min_intensity_for_detection, self.BeadFailReason.TooDim), 
            (too_high, self.BeadFailReason.TooBright), 
            ]
        return regions.apply(lambda row: "".join(code.value if fails(row) else "" for fails, code in invalidation_label_pairs), axis=1)

    def _segment_image(self, img: np.ndarray) -> Tuple[np.ndarray, int]:
        return ndi.label(img > self.min_intensity_for_segmentation)    

    class BeadFailReason(Enum):
        """Why a fiducial bead ROI may be passed discarded, not to be used for drift correction"""
        OutOfBoundsZ = "z"
        OutOfBoundsY = "y"
        OutOfBoundsX = "x"
        TooBig = "s"
        TooDim = "i"
        TooBright = "I"
