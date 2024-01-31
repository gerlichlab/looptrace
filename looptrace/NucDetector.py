# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import os
from pathlib import Path
from typing import *

import dask.array as da
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import expand_labels, find_boundaries, relabel_sequential
from skimage.measure import regionprops_table
from skimage.transform import rescale
from skimage.morphology import remove_small_objects
import tqdm

from looptrace import image_io
from looptrace.numeric_types import NumberLike
from looptrace.wrappers import phase_xcor
from looptrace.Drifter import COARSE_DRIFT_TABLE_COLUMNS, generate_drift_function_arguments__coarse_drift_only


class NucDetector:
    '''
    Class for handling generation and detection of e.g. nucleus images.
    '''
    def __init__(self, image_handler):
        self.image_handler = image_handler

    CLASSES_KEY = "nuc_classes"
    DETECTION_METHOD_KEY = "nuc_method"
    IMAGES_KEY = "nuc_images"
    MASKS_KEY = "nuc_masks"
    KEY_3D = "nuc_3d"

    @property
    def channel(self) -> int:
        return self.image_handler.nuclei_channel

    @property
    def classify_mitotic(self) -> bool:
        return self.config.get("nuc_mitosis_class", False)

    @property
    def config(self) -> Mapping[str, Any]:
        return self.image_handler.config

    @property
    def do_in_3d(self) -> bool:
        return self.config.get(self.KEY_3D, False)

    @property
    def drift_correction_file__coarse(self) -> Path:
        return self.image_handler.get_dc_filepath(prefix="nuclei", suffix="_coarse.csv")

    @property
    def drift_correction_file__fine(self) -> Path:
        return self.image_handler.get_dc_filepath(prefix="nuclei", suffix="_fine.csv")

    @property
    def ds_xy(self) -> int:
        return self.config["nuc_downscaling_xy"]

    @property
    def ds_z(self) -> int:
        if self.do_in_3d:
            return self.config["nuc_downscaling_z"]
        raise NotImplementedError("3D nuclei detection is off, so downscaling in z (ds_z) is undefined!")

    @property
    def images(self) -> List[da.core.Array]:
        imgs = self.image_handler.images[self.input_name]
        if len(imgs) != len(self.pos_list):
            raise Exception(f"{len(imgs)} images and {len(self.pos_list)} positions; these should be equal!")
        exp_shape_len = 4 # (ch, z, y, x) -- no time dimension since only 1 timepoint's imaged for nuclei.
        bad_images = {p: i.shape for p, i in zip(self.pos_list, imgs) if len(i.shape) != exp_shape_len}
        if bad_images:
            raise Exception(f"{len(bad_images)} images with shape length not equal to {exp_shape_len}: {bad_images}")
        return imgs

    def iter_pos_img_pairs(self) -> Iterable[Tuple[str, np.ndarray]]:
        for i, pos in enumerate(self.pos_list):
            yield pos, self.image_handler.images[self.IMAGES_KEY][i]

    def iter_images(self) -> Iterable[np.ndarray]:
        return (img for _, img in self.iter_pos_img_pairs())

    @property
    def segmentation_method(self) -> str:
        return self.config[self.DETECTION_METHOD_KEY]

    @property
    def min_size(self) -> int:
        return self.config["nuc_min_size"]

    @property
    def input_name(self) -> str:
        return self.config["nuc_input_name"]

    @property
    def nuc_classes_path(self) -> Path:
        return self._get_img_save_path(self.CLASSES_KEY)
    
    @property
    def nuc_images_path(self) -> Path:
        return self._get_img_save_path(self.IMAGES_KEY)
    
    @property
    def nuc_masks_path(self) -> Path:
        return self._get_img_save_path(self.MASKS_KEY)
    
    @property
    def pos_list(self) -> List[str]:
        return self.image_handler.image_lists[self.input_name]

    @property
    def reference_frame(self) -> int:
        return self.config["nuc_ref_frame"]

    def _get_img_save_path(self, name: str) -> Path:
        return Path(self.image_handler.image_save_path) / name

    def gen_nuc_images(self):
        '''
        Saves 2D/3D (defined in config) images of the nuclear channel into image folder for later analysis.
        '''
        nuc_slice = self.config.get("nuc_slice", -1)
        if self.do_in_3d:
            axes = ("z", "y", "x")
            prep = lambda img: img
        else:
            axes = ("y", "x")
            # TODO: encode better the meaning of this sentinel for nuc_slice, and document it (i.e., -1 appears to be max-projection).
            # See: https://github.com/gerlichlab/looptrace/issues/244
            prep = (lambda img: da.max(img, axis=0)) if nuc_slice == -1 else (lambda img: img[nuc_slice])
        
        print("Generating nuclei images...")
        name_img_pairs = [(pos_name, prep(self.images[i][self.channel]).compute()) for i, pos_name in tqdm.tqdm(enumerate(self.pos_list))]
        print("Saving nuclei images...")
        if nuc_slice == -1:
            image_io.nuc_multipos_single_time_max_z_proj_zarr(name_img_pairs, root_path=self.nuc_images_path, dtype=np.uint16)
        else:
            for pos_name, subimg in tqdm.tqdm(name_img_pairs):
                # TODO: replace this dimensionality hack with a cleaner solution to zarr writing.
                # See: https://github.com/gerlichlab/looptrace/issues/245
                image_io.single_position_to_zarr(
                    subimg, 
                    path=self.nuc_images_path, 
                    name=self.IMAGES_KEY, 
                    pos_name=pos_name, 
                    axes=axes, 
                    dtype=np.uint16, 
                    chunk_split=(1,1),
                    # TODO: reactivate if using netcdf-java or similar. #127
                    # compressor=numcodecs.Zlib(),
                    )
    
    def segment_nuclei(self) -> Path:
        '''
        Runs nucleus segmentation using nucleus segmentation algorithm defined in ip functions.
        Dilates a bit and saves images.
        '''
        if self.IMAGES_KEY not in self.image_handler.images:
            print(f"{self.IMAGES_KEY} doesn't yet exist as key for subset of images; generating...")
            self.gen_nuc_images()
            print("Re-reading images...")
            self.image_handler.read_images()
        if self.segmentation_method == "threshold":
            return self.segment_nuclei_threshold()
        else:
            return self.segment_nuclei_cellpose()
    
    def segment_nuclei_threshold(self) -> Path:
        for pos, img in self.iter_pos_img_pairs():
            # TODO: need to make this accord with the structure of saved images in segment_nuclei_cellpose.
            # TODO: need to handle whether nuclei images can have more than 1 timepoint (nontrivial time dimension).
            # See: https://github.com/gerlichlab/looptrace/issues/243
            img = img[0, self.channel, ::self.ds_z, ::self.ds_xy, ::self.ds_xy]
            img = ndi.gaussian_filter(img, 2)
            mask = img > int(self.config["nuc_threshold"])
            mask = ndi.binary_fill_holes(mask)
            mask = remove_small_objects(mask, min_size=self.min_size).astype(np.uint16)
            mask = ndi.label(mask)[0]
            mask = rescale(expand_labels(mask.astype(np.uint16),self.config["nuc_dilation"]), scale = (self.ds_z, self.ds_xy, self.ds_xy), order=0)
            # TODO: need to adjust axes argument probably.
            # See: https://github.com/gerlichlab/looptrace/issues/245
            image_io.single_position_to_zarr(mask, path=self.nuc_masks_path, name=self.MASKS_KEY, pos_name=pos, axes=('z','y','x'), dtype=np.uint16, chunk_split=(1,1))

    def segment_nuclei_cellpose(self) -> Path:
        '''
        Runs nucleus segmentation using nucleus segmentation algorithm defined in ip functions.
        Dilates a bit and saves images.
        '''     
        method = self.segmentation_method
        diameter = self.config["nuc_diameter"] / self.ds_xy
        if self.do_in_3d:
            scale_for_rescaling = (self.ds_z, self.ds_xy, self.ds_xy)
            def scale_down_img(img_zyx: np.ndarray) -> np.ndarray:
                assert len(img_zyx.shape) == 3, f"Bad shape for alleged 3D image: {img_zyx.shape}"
                return img_zyx[::self.ds_z, ::self.ds_xy, ::self.ds_xy]
            get_masks = lambda imgs: _nuc_segmentation_cellpose_3d(imgs, diameter=diameter, model_type=method, anisotropy=self.config["nuc_anisotropy"])
            ome_zarr_axes = ("p", "z", "y", "x")
        else:
            scale_for_rescaling = (self.ds_xy, self.ds_xy)
            def scale_down_img(img_zyx: np.ndarray) -> np.ndarray:
                assert len(img_zyx.shape) == 2, f"Bad shape for alleged 3D image: {img_zyx.shape}"
                return img_zyx[::self.ds_xy, ::self.ds_xy]
            get_masks = lambda imgs: _nuc_segmentation_cellpose_2d(imgs, diameter=diameter, model_type=method)
            ome_zarr_axes = ("p", "y", "x")
        
        nuc_min_size = self.min_size / np.prod(scale_for_rescaling)
        
        print("Extracting nuclei images...")
        nuc_imgs = [np.array(scale_down_img(img)) for img in tqdm.tqdm(self.iter_images())]

        print(f"Running nuclear segmentation using CellPose {method} model and diameter {diameter}.")
        # Remove under-segmented nuclei and clean up after getting initial masks.
        masks = [remove_small_objects(arr, min_size=nuc_min_size) for arr in get_masks(nuc_imgs)]
        masks = [relabel_sequential(arr)[0] for arr in masks]

        if self.classify_mitotic:
            print(f"Detecting mitotic cells on top of CellPose nuclei...")
            masks, mitotic_idx = zip(*[_mitotic_cell_extra_seg(np.array(img), mask) for img, mask in zip(nuc_imgs, masks)])

        masks = [rescale(expand_labels(mask.astype(np.uint16), 3), scale=scale_for_rescaling, order=0) for mask in masks]

        print("Saving segmentations...")
        self.image_handler.images[self.MASKS_KEY] = masks
        # TODO: need to adjust axes argument probably.
        # See: https://github.com/gerlichlab/looptrace/issues/245
        image_io.images_to_ome_zarr(images=masks, path=self.nuc_masks_path, name=self.MASKS_KEY, axes=ome_zarr_axes, dtype=np.uint16, chunk_split=(1, 1))
        
        if self.classify_mitotic:
            nuc_class = []
            for i, mask in enumerate(masks):
                class_1 = ((mask > 0) & (mask < mitotic_idx[i])).astype(int)
                class_2 = (mask >= mitotic_idx[i]).astype(int)
                nuc_class.append(class_1 + 2*class_2)
            print("Saving classifications...")
            self.image_handler.images[self.CLASSES_KEY] = nuc_class
            # TODO: need to adjust axes argument probably.
            # See: https://github.com/gerlichlab/looptrace/issues/245
            image_io.images_to_ome_zarr(images=nuc_class, path=self.nuc_classes_path, name=self.CLASSES_KEY, axes=ome_zarr_axes, dtype=np.uint16, chunk_split=(1, 1))

        return self.nuc_masks_path

    def coarse_drift_correction_workflow(self) -> Path:
        from joblib import Parallel, delayed
        downsampling = self.config["coarse_drift_downsample"]
        all_args = generate_drift_function_arguments__coarse_drift_only(
            full_pos_list=self.pos_list, 
            pos_list=self.pos_list, 
            reference_images=self.image_handler.images[self.image_handler.reg_input_template], 
            reference_frame=self.config["reg_ref_frame"],
            reference_channel=self.config["reg_ch_template"],
            moving_images=self.images,
            moving_channel=self.config["reg_ch_moving"],
            downsampling=downsampling,
            nuclei_mode=True,
        )
        print("Computing coarse drifts...")
        records = Parallel(n_jobs=-1, prefer='threads')(
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


    def update_masks_after_qc(self, new_mask, original_mask, mask_name, position):
        s = tuple([slice(None, None, 4)] * len(new_mask.ndim))
        if not np.allclose(new_mask[s], original_mask[s]):
            print("Segmentation labels have changed; resaving...")
            nuc_mask = _relabel_nucs(new_mask)
            pos_index = self.image_handler.image_lists[mask_name].index(position)
            self.image_handler.images[mask_name] = nuc_mask.astype(np.uint16)
            # TODO: need to adjust axes argument probably.
            # See: https://github.com/gerlichlab/looptrace/issues/245
            image_io.single_position_to_zarr(
                images=self.image_handler.images[mask_name][pos_index], 
                path = self.nuc_masks_path / position, 
                name=mask_name, 
                axes=('z','y','x') if self.do_in_3d else ('y','x'), 
                dtype=np.uint16, 
                chunk_split=(1,1),
                )
        else:
            print("Nothing to update, as all values are approximately equal")


    def gen_nuc_rois(self):
        # Use if images are not drift corrected, but a drift correction needs to have been calculated using Drifter.
        nuc_rois = []
        nuc_masks = self.image_handler.images[self.MASKS_KEY]

        if self.CLASSES_KEY in self.image_handler.images:
            print('Adding classes.')
            nuc_classes = self.image_handler.images[self.CLASSES_KEY]
        else:
            nuc_classes = [None]*len(nuc_masks)
        
        for i in tqdm.tqdm(range(len(nuc_masks))):

            mask = np.array(nuc_masks[i])
            nuc_class = np.array(nuc_classes[i])

            if mask.ndim == 2:
                nuc_props = pd.DataFrame(regionprops_table(
                    mask, 
                    intensity_image=nuc_class, 
                    properties=['label', 'bbox', 'intensity_mean']
                    )).rename(columns={
                        'bbox-0':'y_min', 
                        'bbox-1':'x_min', 
                        'bbox-2':'y_max', 
                        'bbox-3':'x_max',
                    })
            else:
                nuc_props = pd.DataFrame(regionprops_table(
                    mask, 
                    intensity_image=nuc_class, 
                    properties=['label', 'bbox', 'intensity_mean']
                    )).rename(columns={
                        'bbox-0':'z_min', 
                        'bbox-1':'y_min', 
                        'bbox-2':'x_min', 
                        'bbox-3':'z_max', 
                        'bbox-4':'y_max', 
                        'bbox-5':'x_max',
                    })

            for j, roi in nuc_props.iterrows():
                old_pos = 'P'+str(i+1).zfill(4)
                new_pos = 'P'+str(i+1).zfill(4)+'_'+str(j+1).zfill(4)
                sel_dc = self.image_handler.tables['drift_correction_full_frame'].query('position == @old_pos')
                ref_offset = sel_dc.query('frame == @ref_frame')
                try:
                    Z, Y, X = self.images[i][self.channel].shape[-3:]
                except AttributeError: #Images not loaded for some reason
                    Z = 200
                    Y = nuc_masks[0].shape[-2]
                    X = nuc_masks[0].shape[-1]
                
                for k, dc_frame in sel_dc.iterrows():
                    z_drift_coarse = int(dc_frame['z_px_coarse']) - int(ref_offset['z_px_coarse'])
                    y_drift_coarse = int(dc_frame['y_px_coarse']) - int(ref_offset['y_px_coarse'])
                    x_drift_coarse = int(dc_frame['x_px_coarse']) - int(ref_offset['x_px_coarse'])

                    if nuc_masks[0].ndim == 2:
                        z_min = 0
                        z_max = Z
                    else:
                        z_min = int(roi['z_min'] - z_drift_coarse)
                        z_max = int(roi['z_max'] - z_drift_coarse)
                    y_min = int(roi['y_min'] - y_drift_coarse)
                    y_max = int(roi['y_max'] - y_drift_coarse)
                    x_min = int(roi['x_min'] - x_drift_coarse)
                    x_max = int(roi['x_max'] - x_drift_coarse)

                    #Handling case of ROI extending beyond image edge after drift correction:
                    pad = ((abs(min(0,z_min)),abs(max(0,z_max-Z))),
                            (abs(min(0,y_min)),abs(max(0,y_max-Y))),
                            (abs(min(0,x_min)),abs(max(0,x_max-X))))

                    sz = (max(0,z_min),min(Z,z_max))
                    sy = (max(0,y_min),min(Y,y_max))
                    sx = (max(0,x_min),min(X,x_max))

                    #Create slice object after above corrections:
                    s = (slice(sz[0],sz[1]), 
                        slice(sy[0],sy[1]), 
                        slice(sx[0],sx[1]))

                    nuc_rois.append([old_pos, new_pos, i, roi.name, dc_frame['frame'], self.reference_frame, self.channel, s, pad, z_drift_coarse, y_drift_coarse, x_drift_coarse, 
                                                                                            dc_frame['z_px_fine'], dc_frame['y_px_fine'], dc_frame['x_px_fine'], roi['intensity_mean']])

        nuc_rois = pd.DataFrame(nuc_rois, columns=['orig_position','position', 'orig_pos_index', 'roi_id', 'frame', 'ref_frame', 'ch', 'roi_slice', 'pad', 'z_px_coarse', 'y_px_coarse', 'x_px_coarse', 
                                                                                                'z_px_fine', 'y_px_fine', 'x_px_fine', 'nuc_class'])
        nuc_rois['nuc_class'] = nuc_rois['nuc_class'].round()
        self.nuc_rois = nuc_rois
        self.nuc_rois.to_csv(self.image_handler.out_path('nuc_table.csv'))

    def gen_nuc_rois_prereg(self):
        nuc_rois = []
        nuc_masks = self.image_handler.images[self.MASKS_KEY]
        
        for i in tqdm.tqdm(range(len(nuc_masks))):

            mask = np.array(nuc_masks[i][0,0])
            if mask.shape[0] == 1:
                nuc_props = pd.DataFrame(regionprops_table(mask, properties=['label', 'bbox', 'area'])).rename(columns={
                    'bbox-0':'y_min', 
                    'bbox-1':'x_min', 
                    'bbox-2':'y_max', 
                    'bbox-3':'x_max',
                })
            else:
                nuc_props = pd.DataFrame(regionprops_table(mask, properties=['label', 'bbox', 'area'])).rename(columns={
                    'bbox-0':'z_min', 
                    'bbox-1':'y_min', 
                    'bbox-2':'x_min', 
                    'bbox-3':'z_max', 
                    'bbox-4':'y_max', 
                    'bbox-5':'x_max',
                })

            nuc_props['orig_position'] = self.image_handler.image_lists[self.MASKS_KEY][i]
            nuc_props['position'] = self.image_handler.image_lists[self.MASKS_KEY][i] + '_' + 'P' + nuc_props['label'].apply(str).str.zfill(3)
            nuc_rois.append(nuc_props)

        self.nuc_rois = pd.concat(nuc_rois).reset_index(drop=True)
        self.nuc_rois.to_csv(self.image_handler.out_path('nuc_rois.csv'))


def _mask_to_binary(mask):
    '''Converts masks from nuclear segmentation to masks with 
    single pixel background between separate, neighbouring features.

    Args:
        masks ([np array]): Detected nuclear masks (label image)

    Returns:
        [np array]: Masks with single pixel seperation beteween neighboring features.
    '''
    masks_no_bound = np.where(find_boundaries(mask)>0, 0, mask)
    return masks_no_bound


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
    from skimage.morphology import label, remove_small_objects
    nuc_int = np.mean(nuc_image[nuc_mask > 0])
    mito_nuc = (nuc_image * (nuc_mask == 0)) > 1.5 * nuc_int
    mito_nuc = remove_small_objects(mito_nuc, min_size=100)
    mito_nuc = label(mito_nuc)
    mito_index = np.max(nuc_mask)
    mito_nuc[mito_nuc > 0] += mito_index
    nuc_mask = nuc_mask + mito_nuc
    return nuc_mask, mito_index + 1


def _nuc_segmentation_cellpose_2d(nuc_imgs: Union[List[np.ndarray], np.ndarray], diameter: NumberLike = 150, model_type = 'nuclei'):
    '''
    Runs nuclear segmentation using cellpose trained model (https://github.com/MouseLand/cellpose)

    Args:
        nuc_imgs (ndarray or list of ndarrays): 2D or 3D images of nuclei, expects single channel
    '''
    if not isinstance(nuc_imgs, list):
        if nuc_imgs.ndim > 2:
            nuc_imgs = [np.array(nuc_imgs[i]) for i in range(nuc_imgs.shape[0])] #Force array conversion in case of zarr.

    from cellpose import models
    model = models.CellposeModel(gpu=False, model_type=model_type)
    masks = model.eval(nuc_imgs, diameter=diameter, channels=[0,0], net_avg=False, do_3D=False)[0]
    return masks


def _nuc_segmentation_cellpose_3d(nuc_imgs: Union[List[np.ndarray], np.ndarray], diameter: NumberLike = 150, model_type: str = 'nuclei', anisotropy: NumberLike = 2):
    '''
    Runs nuclear segmentation using cellpose trained model (https://github.com/MouseLand/cellpose)

    Args:
        nuc_imgs (ndarray or list of ndarrays): 2D or 3D images of nuclei, expects single channel
    '''
    if not isinstance(nuc_imgs, list):
        if nuc_imgs.ndim > 3:
            nuc_imgs = [np.array(nuc_imgs[i]) for i in range(nuc_imgs.shape[0])] #Force array conversion in case of zarr.

    from cellpose import models
    model = models.CellposeModel(gpu=True, model_type=model_type, net_avg=False)
    masks = model.eval(nuc_imgs, diameter=diameter, channels=[0, 0], z_axis=0, anisotropy=anisotropy, do_3D=True)[0]
    return masks


def _relabel_nucs(nuc_image):
    from skimage.morphology import label
    out = _mask_to_binary(nuc_image)
    out = label(out)
    return out.astype(nuc_image.dtype)
