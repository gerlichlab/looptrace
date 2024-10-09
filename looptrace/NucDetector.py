# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from enum import Enum
from operator import itemgetter
import logging
from pathlib import Path
from typing import *

import dask.array as da
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import expand_labels, find_boundaries, relabel_sequential
from skimage.transform import rescale
from skimage.morphology import remove_small_objects, label as morph_label
import tqdm

from looptrace import ArrayDimensionalityError, ConfigurationValueError, MissingImagesError, image_io
from looptrace.numeric_types import NumberLike
from looptrace.wrappers import phase_xcor
from looptrace.Drifter import COARSE_DRIFT_TABLE_COLUMNS, generate_drift_function_arguments__coarse_drift_only

__author__ = "Kai Sandvold Beckwith"
__credits__ = ["Kai Sandvold Beckwith", "Vince Reuter"]


class SegmentationMethod(Enum):
    """Encoding of the methods available for nuclei segmentation"""
    CELLPOSE = "cellpose"
    THRESHOLD = "threshold"

    @classmethod
    def from_string(cls, s: str) -> Optional["SegmentationMethod"]:
        lookup = {k: m for m in cls for k in [m.name, m.value]}
        return lookup.get(s)

    @classmethod
    def unsafe_from_string(cls, s: str) -> "SegmentationMethod":
        member = cls.from_string(s)
        if member is None:
            choices = ", ".join(m.value for m in cls)
            raise ConfigurationValueError(f"Cannot parse '{s}' as nuceli segmentation method; choose from: {choices}")
        return member


class NucDetector:
    '''
    Class for handling generation and detection of e.g. nucleus images.
    '''
    def __init__(self, image_handler):
        self.image_handler = image_handler

    # segmentation settings
    DETECTION_METHOD_KEY = "nuc_method"
    KEY_3D = "nuc_3d"

    # image/label subfolder names
    CLASSES_KEY = "nuc_classes"
    MASKS_KEY = "nuc_masks"
    SEGMENTATION_IMAGES_KEY = "nuc_images"

    # other settings
    _Z_SLICE_KEY = "nuc_slice"

    @property
    def channel(self) -> int:
        """The imaging channel with nuclei stain (generally DAPI) signal"""
        return self.image_handler.nuclei_channel

    @property
    def class_images(self) -> Optional[Sequence]:
        return self.image_handler.images.get(self.CLASSES_KEY)

    @property
    def classify_mitotic(self) -> bool:
        """Whether mitotic status of nuclei should also be evaluated and labeled"""
        return self.config.get("nuc_mitosis_class", False)

    @property
    def config(self) -> Mapping[str, Any]:
        return self.image_handler.config

    @property
    def _defines_z_slice(self) -> bool:
        return self._Z_SLICE_KEY in self.config

    @property
    def do_in_3d(self) -> bool:
        """Whether the deftection and segmentation are to be done using z as well as x and y"""
        return self.config.get(self.KEY_3D, False) or self.segmentation_method == SegmentationMethod.THRESHOLD

    @property
    def drift_correction_file__coarse(self) -> Path:
        """Path to the file with coarse drift correction information for nuclei"""
        return self.image_handler.get_dc_filepath(prefix="nuclei", suffix="_coarse.csv")

    @property
    def drift_correction_file__full(self) -> Path:
        """Path to the file with full drift correction information for nuclei"""
        return self.image_handler.get_dc_filepath(prefix="nuclei", suffix="_full.csv")

    @property
    def ds_xy(self) -> int:
        """Downscaling factor for the x and y dimensions; sample every nth pixel."""
        return self.config["nuc_downscaling_xy"]

    @property
    def ds_z(self) -> int:
        """Downscaling factor for the z dimension; sample every nth pixel."""
        if self.do_in_3d:
            return self.config["nuc_downscaling_z"]
        raise NotImplementedError("3D nuclei detection is off, so downscaling in z (ds_z) is undefined!")

    def _get_img_save_path(self, name: str) -> Path:
        return Path(self.image_handler.image_save_path) / name

    @property
    def has_images_for_segmentation(self) -> bool:
        """Whether this nuclei detection manager is prepared for segmentation"""
        try:
            self.images_for_segmentation
        except MissingImagesError:
            return False
        return True

    @property
    def input_images(self) -> list[da.core.Array]:
        """The unprocessed (other than perhaps format conversion) nuclei images

        Returns
        -------
        list of da.core.Array
            A list of dask arrays, each corresponding to an image stack for a particular FOV / position

        Raises
        ------
        looptrace.MissingImagesError
            If the underlying image handler lacks the key for the nuclei images
        looptrace.ArrayDimensionalityError
            If the number of position names doesn't match the number of image stacks, or
            if any of the image stacks isn't 4-dimensional (1 for channel, and 1 each for (z, y, x))
        """
        try:
            all_imgs = self.image_handler.images
        except AttributeError as e:
            self._raise_missing_images_error(e)
        try:
            imgs = all_imgs[self._input_name]
        except KeyError as e:
            raise MissingImagesError(f"No images available ({self._input_name}) as raw input for nuclei segmentation!") from e
        if len(imgs) != len(self.pos_list):
            raise ArrayDimensionalityError(f"{len(imgs)} images and {len(self.pos_list)} positions; these should be equal!")
        exp_shape_len = 4 # (ch, z, y, x) -- no time dimension since only 1 timepoint's imaged for nuclei.
        bad_images = {p: i.shape for p, i in zip(self.pos_list, imgs) if len(i.shape) != exp_shape_len}
        if bad_images:
            item_list_text = ", ".join(f"{p}: {shape}" for p, shape in sorted(bad_images.items(), key=itemgetter(0)))
            raise ArrayDimensionalityError(f"{len(bad_images)} images with bad shape (length not equal to {exp_shape_len}): {item_list_text}")
        return imgs

    @property
    def images_for_segmentation(self) -> Sequence[np.ndarray]:
        try:
            all_imgs = self.image_handler.images
        except AttributeError as e:
            self._raise_missing_images_error(e)
        try:
            return all_imgs[self.SEGMENTATION_IMAGES_KEY]
        except KeyError as e:
            raise MissingImagesError(
                f"No images available ({self.SEGMENTATION_IMAGES_KEY}) as preprocessed input for nuclei segmentation!"
                ) from e

    @property
    def mask_images(self) -> Sequence[np.ndarray]:
        try:
            all_imgs = self.image_handler.images
        except AttributeError as e:
            self._raise_missing_images_error(e)
        try:
            return all_imgs[self.MASKS_KEY]
        except KeyError as e:
            raise MissingImagesError(
                f"No images available ({self.MASKS_KEY}) as nuclear masks!"
                ) from e

    @property
    def min_size(self) -> int:
        return self.config["nuc_min_size"]

    @property
    def nuc_classes_path(self) -> Path:
        return self._get_img_save_path(self.CLASSES_KEY)
        
    @property
    def nuclear_masks_path(self) -> Path:
        return self._get_img_save_path(self.MASKS_KEY)
    
    @property
    def pos_list(self) -> list[str]:
        """List of names for the fields of view (FOVs) in which nuclei images were taken"""
        try:
            image_lists = self.image_handler.image_lists
        except AttributeError as e:
            raise AttributeError("Position names list for nuclei isn't defined when there are no images!") from e
        try:
            return image_lists[self._input_name]
        except KeyError as e:
            raise AttributeError("Position names list for nuclei isn't defined when there are no nuclei images!") from e

    @property
    def segmentation_method(self) -> SegmentationMethod:
        rawval = self.config[self.DETECTION_METHOD_KEY]
        return SegmentationMethod.unsafe_from_string(rawval)

    @property
    def z_slice_for_segmentation(self) -> int:
        if self.do_in_3d:
            raise NotImplementedError("z-slicing isn't allowed when doing nuclear segmentation in 3D!")
        z = self.config.get(self._Z_SLICE_KEY, -1)
        if not isinstance(z, int):
            raise TypeError(f"z-slice for nuclear segmentation isn't an integer, but {type(z).__name__}: {z}")
        return z

    @property
    def _input_name(self) -> str:
        return self.config["nuc_input_name"]

    def iterate_over_pairs_of_position_and_mask_image(self) -> Iterable[Tuple[str, np.ndarray]]:
        return zip(self.pos_list, self.mask_images, strict=True)

    def iterate_over_pairs_of_position_and_segmentation_image(self) -> Iterable[Tuple[str, np.ndarray]]:
        return zip(self.pos_list, self.images_for_segmentation, strict=True)

    @property
    def nuclear_segmentation_images_path(self) -> Path:
        return self._get_img_save_path(self.SEGMENTATION_IMAGES_KEY)

    def _raise_missing_images_error(self, src: BaseException):
        raise MissingImagesError(f"No images available at all; was {type(self).__name__} created without an images folder?") from src

    def generate_images_for_segmentation(self):
        """Save 2D/3D (defined in config) images of the nuclear channel into image folder for later analysis."""
        if self.do_in_3d:
            if self._defines_z_slice:
                print(f"INFO: z-slice for nuclear segmentation is defined but won't be used for method '{self.segmentation_method}'.")
            axes = ("z", "y", "x")
            prep = lambda img: img
        else:
            axes = ("y", "x")
            # TODO: encode better the meaning of this sentinel for nuc_slice, and document it (i.e., -1 appears to be max-projection).
            # See: https://github.com/gerlichlab/looptrace/issues/244
            nuc_slice = self.z_slice_for_segmentation
            prep = (lambda img: da.max(img, axis=0)) if nuc_slice == -1 else (lambda img: img[nuc_slice])
        
        arr_to_numpy = lambda a: a if isinstance(a, np.ndarray) else a.compute()
        name_img_pairs = [
            (pos_name, arr_to_numpy(prep(self.input_images[i][self.channel]))) 
            for i, pos_name in tqdm.tqdm(enumerate(self.pos_list))
        ]
        print("Generating and saving nuclei images...")
        if self.do_in_3d:
            for pos_name, subimg in tqdm.tqdm(name_img_pairs):
                image_io.single_position_to_zarr(
                    images=subimg, 
                    path=self.nuclear_segmentation_images_path, 
                    name=self.SEGMENTATION_IMAGES_KEY, 
                    pos_name=pos_name, 
                    axes=axes, 
                    dtype=np.uint16, 
                    chunk_split=(1,1),
                    # TODO: reactivate if using netcdf-java or similar. #127
                    # compressor=numcodecs.Zlib(),
                    )
        else:
            image_io.nuc_multipos_single_time_max_z_proj_zarr(name_img_pairs, root_path=self.nuclear_segmentation_images_path, dtype=np.uint16)
    
    def segment_nuclei(self) -> Path:
        '''
        Runs nucleus segmentation using nucleus segmentation algorithm defined in ip functions.
        Dilates a bit and saves images.
        '''
        if not self.has_images_for_segmentation:
            print(f"Images for segmentation don't yet exist; generating...")
            self.generate_images_for_segmentation()
            print("Re-reading images...")
            self.image_handler.read_images()
        if self.segmentation_method == SegmentationMethod.CELLPOSE:
            return self.segment_nuclei_cellpose()
        elif self.segmentation_method == SegmentationMethod.THRESHOLD:
            return self.segment_nuclei_threshold()
        else:
            raise Exception(f"Unknown segmentation method: {self.segmentation_method}")
    
    def segment_nuclei_threshold(self) -> Path:
        for pos, img in self.iterate_over_pairs_of_position_and_segmentation_image():
            # TODO: need to make this accord with the structure of saved images in segment_nuclei_cellpose.
            # TODO: need to handle whether nuclei images can have more than 1 timepoint (nontrivial time dimension).
            # See: https://github.com/gerlichlab/looptrace/issues/243
            img = img[::self.ds_z, ::self.ds_xy, ::self.ds_xy]
            img = ndi.gaussian_filter(img, 2)
            mask = img > int(self.config["nuc_threshold"])
            mask = ndi.binary_fill_holes(mask)
            mask = remove_small_objects(mask, min_size=self.min_size).astype(np.uint16)
            mask = ndi.label(mask)[0]
            mask = rescale(expand_labels(mask.astype(np.uint16),self.config["nuc_dilation"]), scale=(self.ds_z, self.ds_xy, self.ds_xy), order=0)
            # TODO: need to adjust axes argument probably.
            # See: https://github.com/gerlichlab/looptrace/issues/245
            bit_depth: image_io.PixelArrayBitDepth = image_io.PixelArrayBitDepth.unsafe_for_array(mask)
            logging.info(f"Saving nuclear masks with bit depth: {bit_depth}")
            image_io.single_position_to_zarr(
                images=mask, 
                path=self.nuclear_masks_path, 
                name=self.MASKS_KEY, 
                pos_name=pos, 
                axes=("z","y","x"), 
                dtype=bit_depth.value, 
                chunk_split=(1,1),
            )

    def segment_nuclei_cellpose(self) -> Path:
        '''
        Runs nucleus segmentation using nucleus segmentation algorithm defined in ip functions.
        Dilates a bit and saves images.
        '''     
        diameter = self.config["nuc_diameter"] / self.ds_xy
        if self.do_in_3d:
            scale_for_rescaling = (self.ds_z, self.ds_xy, self.ds_xy)
            def scale_down_img(img_zyx: np.ndarray) -> np.ndarray:
                assert len(img_zyx.shape) == 3, f"Bad shape for alleged 3D image: {img_zyx.shape}"
                return img_zyx[::self.ds_z, ::self.ds_xy, ::self.ds_xy]
            get_masks = lambda imgs: _nuc_segmentation_cellpose_3d(imgs, diameter=diameter, anisotropy=self.config["nuc_anisotropy"])
            zarr_axes = ("z", "y", "x")
        else:
            scale_for_rescaling = (self.ds_xy, self.ds_xy)
            def scale_down_img(img_zyx: np.ndarray) -> np.ndarray:
                assert len(img_zyx.shape) == 2, f"Bad shape for alleged 3D image: {img_zyx.shape}"
                return img_zyx[::self.ds_xy, ::self.ds_xy]
            get_masks = lambda imgs: _nuc_segmentation_cellpose_2d(imgs, diameter=diameter)
            zarr_axes = ("y", "x")
        
        nuc_min_size = self.min_size / np.prod(scale_for_rescaling)
        
        print("Extracting nuclei images...")
        name_img_pairs: list[tuple[str, np.ndarray]] = [
            (fov_name, np.array(scale_down_img(img))) 
            for fov_name, img in tqdm.tqdm(self.iterate_over_pairs_of_position_and_segmentation_image())
        ]

        print(f"Running nuclear segmentation using CellPose and diameter {diameter}.")
        # Remove under-segmented nuclei and clean up after getting initial masks.
        masks = [remove_small_objects(arr, min_size=nuc_min_size) for arr in get_masks([img for _, img in name_img_pairs])]
        assert len(masks) == len(name_img_pairs), f"{len(name_img_pairs)} pair(s) of name and image, but {len(masks)} mask(s)"
        name_img_mask_trios: list[tuple[str, np.ndarray, np.ndarray]] = [
            (name, img, relabel_sequential(arr)[0]) 
            for (name, img), arr in zip(name_img_pairs, masks)
        ]

        name_mask_pairs: list[tuple[str, np.ndarray]] = []
        if self.classify_mitotic:
            print(f"Detecting mitotic cells on top of CellPose nuclei...")
            name_mitoindex_pairs: list[tuple[str, int]] = []
            for name, img, mask in name_img_mask_trios:
                curr_mask, curr_idx = _mitotic_cell_extra_seg(np.array(img), mask)
                name_mask_pairs.append((name, curr_mask))
                name_mitoindex_pairs.append((name, curr_idx))
        else:
            name_mask_pairs = [(name, mask) for name, _, mask in name_img_mask_trios]

        name_mask_pairs = [
            (name, rescale(expand_labels(mask.astype(np.uint16), 3), scale=scale_for_rescaling, order=0)) 
            for name, mask in name_mask_pairs
        ]

        self.image_handler.images[self.MASKS_KEY] = [m for _, m in name_mask_pairs]
        saving_prefix = "Overwriting existing nuclear segmentations" if self.nuclear_masks_path.exists() else "Saving nuclear segmentations"
        print(f"{saving_prefix}: {self.nuclear_masks_path}")
        # TODO: need to adjust axes argument probably.
        # See: https://github.com/gerlichlab/looptrace/issues/247
        image_io.images_to_ome_zarr(
            name_image_pairs=name_mask_pairs, 
            path=self.nuclear_masks_path, 
            data_name=self.MASKS_KEY, 
            axes=zarr_axes, 
            chunk_split=(1, 1),
        )
        
        if self.classify_mitotic:
            name_class_pairs: list[tuple[str, np.ndarray]] = []
            for (name_mask, mask), (name_mito, mitoidx) in zip(name_mask_pairs, name_mitoindex_pairs):
                if name_mask != name_mito:
                    raise RuntimeError(
                        f"Nuclear mask name doesn't match mitotic index name. {name_mask} != {name_mito} . Maintenance of parallel named lists failed!"
                    )
                class_1 = ((mask > 0) & (mask < mitoidx)).astype(int)
                class_2 = (mask >= mitoidx).astype(int)
                name_class_pairs.append((name_mask, class_1 + 2*class_2))
            self.image_handler.images[self.CLASSES_KEY] = [c for _, c in name_class_pairs]
            # TODO: need to adjust axes argument probably.
            # See: https://github.com/gerlichlab/looptrace/issues/247
            image_io.images_to_ome_zarr(
                name_image_pairs=name_class_pairs, 
                path=self.nuc_classes_path, 
                data_name=self.CLASSES_KEY, 
                axes=zarr_axes, 
                chunk_split=(1, 1),
            )

        return self.nuclear_masks_path

    def coarse_drift_correction_workflow(self) -> Path:
        from joblib import Parallel, delayed
        downsampling = self.config["coarse_drift_downsample"]
        # TODO: check that the structure of the images (moving and template) matches the indexing in this function w.r.t. (t, c, z, y, x). See #241.
        all_args = generate_drift_function_arguments__coarse_drift_only(
            full_pos_list=self.pos_list, 
            pos_list=self.pos_list, 
            reference_images=self.image_handler.drift_correction_reference_images, 
            reference_timepoint=self.image_handler.drift_correction_reference_timepoint,
            reference_channel=self.image_handler.drift_correction_reference_channel,
            moving_images=self.input_images,
            moving_channel=self.image_handler.drift_correction_moving_channel,
            downsampling=downsampling,
            nuclei_mode=True,
        )
        print("Computing coarse drifts...")
        records = Parallel(n_jobs=-1)(
            delayed(lambda p, t, ref_ds, mov_ds: (t, p) + tuple(phase_xcor(ref_ds, mov_ds) * downsampling))(*args) 
            for args in all_args
            )
        try:
            coarse_drifts = pd.DataFrame(records, columns=COARSE_DRIFT_TABLE_COLUMNS)
        except ValueError: # most likely if element count of one or more rows doesn't match column count
            print(f"Example record (below):\n{records[0]}")
            raise
        outfile = self.drift_correction_file__coarse
        print(f"Writing coarse drifts: {outfile}")
        coarse_drifts.to_csv(outfile)
        return outfile


def _mitotic_cell_extra_seg(nuc_image, nuc_mask):
    '''Performs additional mitotic cell segmentation on top of an interphase segmentation (e.g. from CellPose).
    Assumes mitotic cells are brighter, unsegmented objects in the image.

    Args:
        nuc_image ([nD numpy array]): nuclei image
        nuc_mask ([nD numpy array]): labeled nuclei from nuclei image

    Returns:
        nuc_mask ([nD numpy array]): labeled nuclei with mitotic cells added
        mito_index+1 (int): the first index of the mitotic cells in the returned nuc_mask

    '''
    nuc_int = np.mean(nuc_image[nuc_mask > 0])
    mito_nuc = (nuc_image * (nuc_mask == 0)) > 1.5 * nuc_int
    mito_nuc = remove_small_objects(mito_nuc, min_size=100)
    mito_nuc = morph_label(mito_nuc)
    mito_index = np.max(nuc_mask)
    mito_nuc[mito_nuc > 0] += mito_index
    nuc_mask = nuc_mask + mito_nuc
    return nuc_mask, mito_index + 1


def _nuc_segmentation_cellpose_2d(nuc_imgs: Union[List[np.ndarray], np.ndarray], diameter: NumberLike = 150):
    '''
    Runs nuclear segmentation using cellpose trained model (https://github.com/MouseLand/cellpose)

    Args:
        nuc_imgs (ndarray or list of ndarrays): 2D or 3D images of nuclei, expects single channel
    '''
    if not isinstance(nuc_imgs, list):
        if nuc_imgs.ndim > 2:
            nuc_imgs = [np.array(nuc_imgs[i]) for i in range(nuc_imgs.shape[0])] #Force array conversion in case of zarr.

    from cellpose import models
    model = models.CellposeModel(gpu=False, model_type="nuclei")
    masks = model.eval(nuc_imgs, diameter=diameter, channels=[0, 0], net_avg=False, do_3D=False)[0]
    return masks


def _nuc_segmentation_cellpose_3d(nuc_imgs: Union[List[np.ndarray], np.ndarray], diameter: NumberLike = 150, anisotropy: NumberLike = 2):
    '''
    Runs nuclear segmentation using cellpose trained model (https://github.com/MouseLand/cellpose)

    Args:
        nuc_imgs (ndarray or list of ndarrays): 2D or 3D images of nuclei, expects single channel
    '''
    if not isinstance(nuc_imgs, list):
        if nuc_imgs.ndim > 3:
            nuc_imgs = [np.array(nuc_imgs[i]) for i in range(nuc_imgs.shape[0])] #Force array conversion in case of zarr.

    from cellpose import models
    model = models.CellposeModel(gpu=True, model_type="nuclei", net_avg=False)
    masks = model.eval(nuc_imgs, diameter=diameter, channels=[0, 0], z_axis=0, anisotropy=anisotropy, do_3D=True)[0]
    return masks
