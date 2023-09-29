# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import logging
import glob
import os
import re
from typing import *
import numpy as np
import pandas as pd

import scipy.ndimage as ndi
from scipy.spatial.distance import squareform, pdist
from scipy.stats import trim_mean
from skimage.segmentation import clear_border, find_boundaries, expand_labels
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import white_tophat, ball, remove_small_objects
from skimage.measure import regionprops_table

from looptrace.wrappers import phase_cross_correlation

logger = logging.getLogger()


def rois_from_csv(path):
    rois = pd.read_csv(path, index_col=0)
    print('Loaded existing ROIs from ', path)
    return rois

def rois_from_imagej(roi_folder_path, template = '.zip', crop_size_z = 16, roi_scale = 0.5):
    from .image_io import all_matching_files_in_subfolders

    roi_files = all_matching_files_in_subfolders(roi_folder_path, template)
    all_roi_coords = []
    for file_id, roi_path in enumerate(roi_files):
        position = re.search('W[0-9]*', roi_path).group(0)
        rois=read_roi_zip(roi_path)
        rois=[rois[k] for k in list(rois)]
        for roi_id, roi in enumerate(rois):
            z_min = int(roi['position']['slice']-1-crop_size_z//2)
            z_max = int(roi['position']['slice']-1+crop_size_z//2)
            y_min = int((roi['top']-1)/roi_scale)
            y_max = int(y_min + roi['height']/roi_scale)
            x_min = int((roi['left']-1)/roi_scale)
            x_max = int(x_min + roi['width']/roi_scale)
            all_roi_coords.append([file_id, roi_path, position, roi_id, z_min, z_max, y_min, y_max, x_min, x_max])
    roi_table = pd.DataFrame(all_roi_coords, columns=[   'file_id',
                                                            'roi_path',
                                                            'position', 
                                                            'roi_id',
                                                            'z_min',
                                                            'z_max',
                                                            'y_min',
                                                            'y_max',
                                                            'x_min',
                                                            'x_max',
                                                            ])
    print('Loaded ImageJ ROIs from ', roi_folder_path)
    return roi_table

def roi_to_napari_points(roi_table, position):
    '''Convert roi from looptrace roi table to points to see in napari.

    Args:
        roi_table (DataFrame): ROI data found in looptrace pipeline
        position (str): Positional identifier

    Returns:
        roi_shapes (ndarray): Numpy array of shape NX4, with the 4 volumns (frame, z, y, x)
        roi_props (dict): A dict of the roi_ids.
    '''

    rois_at_pos = roi_table[roi_table['position']==position]
    roi_shapes = []
    for i, roi in rois_at_pos.iterrows():
        try:
            roi_shape = [roi['frame'], roi['zc'], roi['yc'], roi['xc']]
        except KeyError:
            roi_shape = [0, roi['zc'], roi['yc'], roi['xc']]
        roi_shapes.append(roi_shape)
    roi_shapes = np.array(roi_shapes)
    roi_props = {'roi_id': rois_at_pos['roi_id_pos'].values}
    return roi_shapes, roi_props

def update_roi_points(point_layer, roi_table, position, downscale):
    '''Takes (possibly) updated points from napari and converts them into ROI table format for looptrace

    Args:
        point_layer (ndarray): The points returned by napari
        roi_table (DataFrame): ROI table to update
        position (str): Positional identifier
        downscale (int): Downscale factor, if downscaling was used when viewing with napari

    Returns:
        DataFrame: Updated ROI table for looptrace
    '''

    rois = roi_table.copy()
    new_rois = pd.DataFrame(point_layer.data*downscale, columns=['frame','zc','yc','xc'])
    new_rois.index.name = 'roi_id_pos'
    new_rois = new_rois.reset_index()
    new_rois['position'] = position
    new_rois['ch'] = rois.iloc[0]['ch']

    rois = rois.drop(rois[rois['position']==position].index)
    return pd.concat([rois, new_rois]).sort_values('position').reset_index(drop=True)


def filter_rois_in_nucs(rois, nuc_label_img, new_col='nuc_label', nuc_drifts=None, nuc_target_frame=None, spot_drifts=None):
    '''Check if a spot is in inside a segmented nucleus.

    Args:
        rois (DataFrame): ROI table to check
        nuc_label_img (list): 2D/3D label images, where 0 is outside nuclei and >0 inside
        pos_list (list): List of all the positions (str) to check
        new_col (str, optional): The name of the new column in the ROI table. Defaults to 'nuc_label'.

    Returns:
        rois (DataFrame): Updated ROI table indicating if ROI is inside nucleus or not.
    '''

    new_rois = rois.copy()
    print(nuc_label_img.shape)
    def spot_in_nuc(row, nuc_label_img):
        try:
            if nuc_label_img.shape[-3] == 1:
                spot_label = int(nuc_label_img[0, int(row['yc']), int(row['xc'])])
            else:
                spot_label = int(nuc_label_img[int(row['zc']),int(row['yc']), int(row['xc'])])
        except IndexError as e: #If due to drift spot is outside frame.
            spot_label = 0
            print(e)
        #print(spot_label)
        return spot_label

    try:
        new_rois.drop(columns=[new_col], inplace=True)
    except KeyError:
        pass
    #print(rois, nuc_drifts)
    if nuc_drifts is not None:
        rois_shifted = new_rois.copy()
        shifts = []
        for i, row in rois_shifted.iterrows():
            drift_target = nuc_drifts[(nuc_drifts['position'] == row['position']) & (nuc_drifts['frame'] == nuc_target_frame)][['z_px_course', 'y_px_course', 'x_px_course']].to_numpy()
            drift_roi = spot_drifts[(spot_drifts['position'] == row['position']) & (spot_drifts['frame'] == row['frame'])][['z_px_course', 'y_px_course', 'x_px_course']].to_numpy()
            shift = drift_target - drift_roi
            shifts.append(shift[0])
        shifts = pd.DataFrame(shifts, columns=['z','y','x'])
        rois_shifted[['zc', 'yc', 'xc']] = rois_shifted[['zc', 'yc', 'xc']].to_numpy() - shifts[['z','y','x']].to_numpy()

        new_rois.loc[:,new_col] = rois_shifted.apply(spot_in_nuc, nuc_label_img=nuc_label_img, axis=1)
    
    else:
        new_rois.loc[:,new_col] = new_rois.apply(spot_in_nuc, nuc_label_img=nuc_label_img, axis=1)

    return new_rois


def subtract_crosstalk(source, bleed, threshold=500):
    shift = drift_corr_course(source, bleed, downsample=1)
    bleed = ndi.shift(bleed, shift=shift, order=1)
    mask = bleed > threshold
    ratio = np.average(source[mask] / bleed[mask])
    print(ratio)
    out = np.clip(source - (ratio * bleed), a_min=0, a_max=None)
    return out, bleed


def pad_to_shape(arr, shape, mode='constant'):
    '''
    Pads an array with fill to a given shape (list or tuple).
    Shape must be of length equal to array ndim.
    Adds on both sides as far as possible, then at end if missing one.
    Returns padded array.
    '''
    if 0 in arr.shape:
        return np.zeros(shape)
    p = np.subtract(shape,arr.shape)//2
    try:
        assert all(p>=0), 'Cannot pad to smaller than original size. Cropping instead.'
    except AssertionError:
        exp_shape=tuple([np.max((i,j)) for i,j in zip(arr.shape,shape)])
        arr=pad_to_shape(arr, exp_shape, mode)
        arr=crop_to_shape(arr,shape)
        return arr
    ps = tuple((n,n) for n in p)
    arr=np.pad(arr,ps,mode)
    if arr.shape == shape:
        return arr
    else:
        p=np.subtract(shape,arr.shape)
        ps = tuple((0,n) for n in p)
        arr=np.pad(arr,ps,mode)
        return arr
    
def crop_to_shape(arr, shape):
    '''
    Crops an array to a given shape (list or tuple) at center.
    Shape must be same length as array ndim.
    Crops on both sides as far as possible, crops rest at start of each ndim.
    Returns cropped array.
    '''
           
    new_s=np.subtract(arr.shape,shape)//2
    assert all(new_s>=0), 'Cannot crop to larger than original size.'
    s=tuple([slice(None,None) if s==0 else slice(s,-s) for s in new_s])
    arr=arr[s]
    if arr.shape == shape:
        return arr
    else:
        new_s=np.subtract(arr.shape,shape)
        s=tuple([slice(s,None) for s in new_s])
        return arr[s]
    
def crop_at_pos(arr, tl_pos, size):
    '''
    Crops an nd array to given size from a position in the
    corner closest to the origin.
    Enforce int type.
    '''
    s=tuple([slice(int(pos),int(pos+si)) for pos,si in zip(tl_pos,size)])

    return arr[s]

def crop_to_center(arr, center, size):
    #Crop array to size centered at center position.
    s=tuple([slice(min(0,int(c-s//2)),max(int(c+s//2), arr_s)) for c, s, arr_s in zip(center,size,arr.shape)])
    return arr[s]
    
def find_center_of_embryo(embryo_stack):
    #Find center of embryo:
    blur = ndi.gaussian_filter(embryo_stack[::4,::4,::4], 5)
    thresh = np.min(blur)+30
    binary = blur > thresh
    labels, n_labels = ndi.label(remove_small_objects(binary, min_size=5000))
    props = regionprops_table(labels, properties=['bbox', 'centroid'])
    zc = ((props['bbox-3']-props['bbox-0'])//2+props['bbox-0'])*4
    yc = ((props['bbox-4']-props['bbox-1'])//2+props['bbox-1'])*4
    xc = ((props['bbox-5']-props['bbox-2'])//2+props['bbox-2'])*4
    return (zc, yc, xc)

def center_crop_embryo(embryo_stack, size, center=None):
    #Find center of embryo and crop image to specific size, padding if needed.
    if center is None:
        center = find_center_of_embryo(embryo_stack)
    out = crop_to_center(embryo_stack, center, size)
    if out.shape != tuple(size):
        out = pad_to_shape(out, size)
    return out

def detect_spots(input_img, spot_threshold=20, min_dist=None):
    '''Spot detection by difference of gaussian filter
    #TODO: Do not use hard-coded sigma values

    Args:
        img (ndarray): Input 3D image
        spot_threshold (int): Threshold to use for spots. Defaults to 20.

    Returns:
        spot_props (DataFrame): The centroids and roi_IDs of the spots found. 
        img (ndarray): The DoG filtered image used for spot detection.
    '''
    img = white_tophat(image=input_img, footprint=ball(2))
    img = gaussian(img, 0.8) - gaussian(img, 1.3)
    img = img / gaussian(input_img, 3)
    img = (img - np.mean(img)) / np.std(img)
    labels, num_spots = ndi.label(img > spot_threshold)
    labels = expand_labels(labels, 10)
    
    #Make a DataFrame with the ROI info
    spot_props=pd.DataFrame(regionprops_table(label_image = labels, 
                                        intensity_image = input_img,
                                        properties=('label','centroid_weighted')))
    
    spot_props.drop(['label'], axis=1, inplace=True)
    spot_props.rename(columns={'centroid_weighted-0': 'zc',
                                        'centroid_weighted-1': 'yc',
                                        'centroid_weighted-2': 'xc'},
                        inplace = True)

    if min_dist:
        dists = squareform(pdist(spot_props[['zc', 'yc', 'xc']].to_numpy(), metric='euclidean'))
        idx = np.nonzero(np.triu(dists < min_dist, k=1))[1]
        spot_props = spot_props.drop(idx)
        spot_props = spot_props.reset_index(drop=True)

    spot_props.rename(columns={'index':'roi_id'},
                                inplace = True)

    #print(f'Found {len(spot_props)} spots.', end = ' ')
    return spot_props, img

def detect_spots_int(input_img, spot_threshold=500, expand_px = 1, min_dist=None):
    '''Spot detection by difference of gaussian filter
    #TODO: Do not use hard-coded sigma values

    Args:
        img (ndarray): Input 3D image
        spot_threshold (int): Threshold to use for spots. Defaults to 500.

    Returns:
        spot_props (DataFrame): The centroids and roi_IDs of the spots found. 
        img (ndarray): The DoG filtered image used for spot detection.
    '''
    #img = white_tophat(image=input_img, footprint=ball(2))
    struct = ndi.generate_binary_structure(input_img.ndim, 2)
    labels, n_obj = ndi.label(input_img > spot_threshold, structure=struct)
    if n_obj > 1: #Do not need this with area filtering below.
        #pass
        labels = remove_small_objects(labels, min_size=5)
    if expand_px > 0:
        labels = expand_labels(labels, expand_px)
    if np.all(labels == 0): #If there are no labels anymore:
        return pd.DataFrame(columns=['label', 'z_min','y_min','x_min','z_max','y_max','x_max','area','zc','yc','xc']), labels
    else:
        spot_props = regionprops_table(labels, input_img, properties=('label', 'bbox', 'area', 'centroid_weighted'))
        spot_props = pd.DataFrame(spot_props)
        #spot_props = spot_props.query('area > 10')

        spot_props = spot_props.rename(columns={'centroid_weighted-0': 'zc',
                                            'centroid_weighted-1': 'yc',
                                            'centroid_weighted-2': 'xc',
                                            'bbox-0': 'z_min',
                                            'bbox-1': 'y_min',
                                            'bbox-2': 'x_min',
                                            'bbox-3': 'z_max',
                                            'bbox-4': 'y_max',
                                            'bbox-5': 'x_max'})
        
        if min_dist:
            dists = squareform(pdist(spot_props[['zc', 'yc', 'xc']].to_numpy(), metric='euclidean'))
            idx = np.nonzero(np.triu(dists < min_dist, k=1))[1]
            spot_props = spot_props.drop(idx)
        
        spot_props = spot_props.reset_index(drop=True)
        spot_props = spot_props.rename(columns={'index':'roi_id'})

        #print(f'Found {len(spot_props)} spots.', end=' ')
        return spot_props, labels

def roi_center_to_bbox(rois: pd.DataFrame, roi_size: Union[np.ndarray, Tuple[int, int, int]]):
    """Make bounding box coordinates around centers of regions of interest, based on box dimensions."""
    rois['z_min'] = rois['zc'] - roi_size[0]//2
    rois['z_max'] = rois['zc'] + roi_size[0]//2
    rois['y_min'] = rois['yc'] - roi_size[1]//2
    rois['y_max'] = rois['yc'] + roi_size[1]//2
    rois['x_min'] = rois['xc'] - roi_size[2]//2
    rois['x_max'] = rois['xc'] + roi_size[2]//2
    return rois

def generate_bead_rois(t_img, threshold, min_bead_int, bead_roi_px=16, n_points=200, max_size=500):
    '''Function for finding positions of beads in an image based on manually set thresholds in config file.

    Args:
        t_img (3D ndarray): Image
        threshold (float): Threshold for initial bead segmentation
        min_bead_int (float): Secondary filtering of segmented maxima.
        n_points (int): How many bead positions to return

    Returns:
        t_img_maxima: 3XN ndarray of 3D bead coordinates in t_img.
    '''
    roi_px = bead_roi_px//2
    t_img_label,num_labels=ndi.label(t_img>threshold)
    print('Number of unfiltered beads found: ', num_labels)
    t_img_maxima = pd.DataFrame(regionprops_table(t_img_label, t_img, properties=('label', 'centroid', 'max_intensity', 'area')))
    
    cent0, cent1, cent2 = "centroid-0", "centroid-1", "centroid-2"
    t_img_maxima = t_img_maxima[(t_img_maxima[cent0] > roi_px) & (t_img_maxima[cent1] > roi_px) & (t_img_maxima[cent2] > roi_px) & (t_img_maxima['area'] < max_size)].query('max_intensity > @min_bead_int')
    
    centroid_columns = [cent0, cent1, cent2]
    t_img_maxima = t_img_maxima if n_points == -1 else t_img_maxima.sample(n=min(n_points, len(t_img_maxima)), random_state=1)
    return np.round(t_img_maxima[centroid_columns].to_numpy()).astype(int)


def extract_single_bead(
        point: Union[np.ndarray, Iterable[int]], 
        img: np.ndarray, 
        bead_roi_px: int = 16, 
        drift_course: Union[None, np.ndarray, Iterable[int]] = None
        ) -> np.ndarray:
    """
    Extract a cropped region of a single fiducial in an image, optionally including a pre-calucalated course drift to shift the cropped region.

    Parameters
    ----------
    point : np.ndarray or Iterable of int
        Coordinates representing the center of a detected bead
    img : np.ndarray
        An array of values representing an image in which a fiducial bead is detected
    bead_roi_px : int
        The number of pixels for the side length of the bead ROI
    drift_course : None or np.ndarray or Iterable of int
        The coarse-grained drift correction already computed

    Returns
    -------
    np.ndarray
        A numpy array representing the subspace of the given image corresponding to the bead ROI
    """
    roi_px = bead_roi_px // 2
    coords = point if drift_course is None else (x - int(dx) for x, dx in zip(point, drift_course))
    s = tuple([slice(p - roi_px, p + roi_px) for p in coords])
    bead = img[s]
    side_length = 2 * roi_px
    output_shape = (side_length, side_length, side_length)
    # TODO: consider, for provenance, logging a message here, that the bead shape was not as expected, and all-0s is used.
    return np.zeros(output_shape) if bead.shape != output_shape else bead

def downsample_image(img: np.ndarray, step_size: int) -> np.ndarray:
    return np.array(img[tuple(slice(None, None, step_size) for _ in img.shape)])

def drift_corr_course(t_img, o_img, downsample=1):
    '''
    Calculates course and fine 
    drift between two svih5 images by phase cross correlation.

    Parameters
    ----------
    t_path : Path to template image in svih5 format.
    o_path : Path to offset image in svih5 format.

    Returns
    -------
    A list of zyx course drifts and fine drifts (compared to course)

    '''        
    course_drift, _, _ = phase_cross_correlation(downsample_image(t_img), downsample_image(o_img)) * downsample
    return course_drift

def drift_corr_multipoint_cc(t_img, o_img, course_drift, threshold, min_bead_int, n_points=50, upsampling=100):
    '''
    Function for fine scale drift correction. 

    Parameters
    ----------
    t_img : Template image, 2D or 3D ndarray.
    o_img : Offset image, 2D or 3D ndarray.
    threshold : Int, threshold to segment fiducials.
    min_bead_int : Int, minimal value for maxima of fiducials 
    n_points : Int, number of fiducials to use for drift correction. The default is 5.
    upsampling : Int, upsampling grid for subpixel correlation. The default is 100.

    Returns
    -------
    A trimmed mean (default 20% on each side) of the drift for each fiducial.

    '''
    import joblib

    #Label fiducial candidates and find maxima.
    t_img_label,num_labels=ndi.label(t_img>threshold)
    print('Number of unfiltered beads found: ', num_labels)
    t_img_maxima = pd.DataFrame(regionprops_table(t_img_label, t_img, properties=('label', 'centroid', 'max_intensity')))
    t_img_maxima = t_img_maxima.query('max_intensity > @min_bead_int').sample(n=n_points, random_state=1)[['centroid-0', 'centroid-1', 'centroid-2']].to_numpy()
    t_img_maxima = np.round(t_img_maxima).astype(int)

    def extract_and_correlate(point, t_img, o_img, upsampling):

        #Calculate fine scale drift for all selected fiducials.
        s_t = tuple([slice(ind-8, ind+8) for ind in point])
        s_o = tuple([slice(ind-int(shift)-8, ind-int(shift)+8) for (ind, shift) in zip(point, course_drift)])
        t = t_img[s_t]
        o = o_img[s_o]
        
        try:
            shift, _, _ = phase_cross_correlation(t, o, upsample_factor=upsampling)
        except (ValueError, AttributeError):
            shift = np.array([0,0,0])
        return shift

    shifts = joblib.Parallel(n_jobs=-1, prefer='threads')(joblib.delayed(extract_and_correlate)(point, t_img, o_img, upsampling) for point in t_img_maxima)
    shifts = np.array(shifts)

    fine_drift = trim_mean(shifts, proportiontocut=0.2, axis=0)
    print(f'Fine drift: {fine_drift} with untrimmed STD of {np.std(shifts, axis=0)} .')
    #Return the 60% central mean to avoid outliers.
    return fine_drift#, np.std(shifts, axis=0)

def napari_view(img, points=None, downscale=2, axes = 'PTCZYX', point_frame_size = 1, name=None, contrast_limits=(100,10000)):
    import dask.array as da
    import napari

    try:
        channel_axis = axes.index('C')
    except ValueError:
        channel_axis = None
    
    if 'ZYX' in axes:
        img = img[...,::downscale,::downscale,::downscale]
    else:
        img = img[...,::downscale,::downscale]

    if isinstance(img, list):
        img = da.stack(img)

    viewer = napari.view_image(img, channel_axis = channel_axis, name=name)
    if points is not None:
        point_layer = viewer.add_points(points/downscale, 
                                                size=(point_frame_size,15,15,15),
                                                edge_width=3,
                                                edge_width_is_relative=False,
                                                edge_color='red',
                                                face_color='transparent',
                                                n_dimensional=True)
        sel_dim = list(points[0,:]/downscale)
        for dim in range(len(sel_dim)):
            viewer.dims.set_current_step(dim, sel_dim[dim])
    napari.run()

    if points is not None:
        return point_layer

def decon_RL_setup(size_x=8, size_y=8, size_z=8, pz=0., wavelength=.660,
            na=1.46, res_lateral=.1, res_axial=.2):
    '''
    Uses flowdec (https://github.com/hammerlab/flowdec) to perform
    Richardson Lucy deconvolution, using standard settings.
    TODO: Define PSF parameters in input.

    Returns:
        algo: the algorithm for running deconvolution
        kernel: the PSF for running deconvolution
        fd_data: the module for running deconvolution
    '''
    from flowdec import data as fd_data
    from flowdec import restoration as fd_restoration
    from flowdec import psf as fd_psf
    algo = fd_restoration.RichardsonLucyDeconvolver(3).initialize()
    kernel = fd_psf.GibsonLanni(
            size_x=size_x, size_y=size_y, size_z=size_z, pz=pz, wavelength=wavelength,
            na=na, res_lateral=res_lateral, res_axial=res_axial
        ).generate()
    return algo, kernel, fd_data

def decon_RL(img, kernel, algo, fd_data, niter=10):
    '''[summary]

    Args:
        img ([ndarray]): [description]
        kernel ([type]): [description]
        algo ([type]): [description]
        fd_data ([type]): [description]
        niter (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: [description]
    '''
    res = algo.run(fd_data.Acquisition(data=img, kernel=kernel), niter=niter).data
    return res

def nuc_segmentation_cellpose_2d(nuc_imgs, diameter = 150, model_type = 'nuclei'):
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
    masks = model.eval(nuc_imgs, diameter=diameter, channels=[0,0], net_avg=False, do_3D = False)[0]
    return masks

def nuc_segmentation_cellpose_3d(nuc_imgs, diameter = 150, model_type = 'nuclei', anisotropy = 2):
    '''
    Runs nuclear segmentation using cellpose trained model (https://github.com/MouseLand/cellpose)

    Args:
        nuc_imgs (ndarray or list of ndarrays): 2D or 3D images of nuclei, expects single channel
    '''
    from cellpose import models

    if not isinstance(nuc_imgs, list):
        if nuc_imgs.ndim > 3:
            nuc_imgs = [np.array(nuc_imgs[i]) for i in range(nuc_imgs.shape[0])] #Force array conversion in case of zarr.

    model = models.CellposeModel(gpu=True, model_type=model_type, net_avg=False)
    masks = model.eval(nuc_imgs, diameter=diameter, channels=[0,0], z_axis = 0, anisotropy = anisotropy, do_3D=True)[0]
    return masks

def mask_to_binary(mask):
    '''Converts masks from nuclear segmentation to masks with 
    single pixel background between separate, neighbouring features.

    Args:
        masks ([np array]): Detected nuclear masks (label image)

    Returns:
        [np array]: Masks with single pixel seperation beteween neighboring features.
    '''
    masks_no_bound = np.where(find_boundaries(mask)>0, 0, mask)
    return masks_no_bound

def mitotic_cell_extra_seg(nuc_image, nuc_mask):
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
    nuc_int = np.mean(nuc_image[nuc_mask>0])
    mito_nuc = (nuc_image * (nuc_mask == 0)) > 1.5*nuc_int
    mito_nuc = remove_small_objects(mito_nuc, min_size=100)
    mito_nuc = label(mito_nuc)
    mito_index = np.max(nuc_mask)
    mito_nuc[mito_nuc>0] += mito_index
    nuc_mask = nuc_mask + mito_nuc
    return nuc_mask, mito_index+1

def nuc_segmentation_otsu(nuc_image, min_size, exp_bb=-1, clear_edges=True):
    '''
    Detects nuclei based on otsu threshold, labels them and measures
    properties of segmented nuclei. Optionally excludes small nuclei, and calculates
    a larger bounding box based on config file input.
    Input: 3D image of nuclei.
    Returns: Labeled nucleus image, pandas dataframe of nuclear properties.
    
    COULD BE IMPLEMENTED:
    Watershed:
    nuc_image=gaussian(nuc_image, sigma=15)
    thresh_nuc = threshold_otsu(nuc_image)
    nuc_bw = nuc_image > thresh_nuc
    distance = gaussian(ndi.distance_transform_edt(nuc_bw),20)
    local_maxi = peak_local_max(distance, indices=False, exclude_border=0)
    markers = ndi.label(local_maxi)[0]
    nuc_labels = watershed(-distance, markers, mask=nuc_bw)
    '''
    #Blur image for cleaner thresholds
    nuc_image=gaussian(nuc_image, sigma=5)
    thresh_nuc = threshold_otsu(nuc_image)
    print('Nuc threshold:', thresh_nuc)
    #thresh_nuc = 15000
    nuc_labels, num_objects = ndi.label(nuc_image>thresh_nuc)
    
    if clear_edges:
        #Define border mask for object clearing only in x and y, not z
        border_mask=np.zeros(nuc_labels.shape, dtype=bool)
        border_mask[:,1:-1,1:-1]=1
        nuc_labels = clear_border(nuc_labels, mask=border_mask)
    nuc_props=pd.DataFrame(regionprops_table(nuc_labels, nuc_image, 
                                              properties=('label',
                                                          'area',
                                                          'bbox',
                                                          'weighted_centroid',
                                                          )))
    nuc_props=nuc_props[nuc_props['area']>min_size]
    nuc_props=nuc_props.reset_index()
    nuc_labels=nuc_labels*np.isin(nuc_labels, nuc_props['label'])
    
    if exp_bb != -1:
    #Expand bounding box to larger than standard size.
        sz,sy,sx=exp_bb
        nuc_props['lbbox-0'] = (nuc_props['bbox-0']-(sz-(nuc_props['bbox-3']-nuc_props['bbox-0']))//2).clip(lower=0, upper=nuc_image.shape[0])
        nuc_props['lbbox-3'] = (nuc_props['bbox-3']+(sz-(nuc_props['bbox-3']-nuc_props['bbox-0']))//2).clip(lower=0, upper=nuc_image.shape[0])
        nuc_props['lbbox-1'] = (nuc_props['bbox-1']-(sy-(nuc_props['bbox-4']-nuc_props['bbox-1']))//2).clip(lower=0, upper=nuc_image.shape[1])
        nuc_props['lbbox-4'] = (nuc_props['bbox-4']+(sy-(nuc_props['bbox-4']-nuc_props['bbox-1']))//2).clip(lower=0, upper=nuc_image.shape[1])
        nuc_props['lbbox-2'] = (nuc_props['bbox-2']-(sx-(nuc_props['bbox-5']-nuc_props['bbox-2']))//2).clip(lower=0, upper=nuc_image.shape[2])
        nuc_props['lbbox-5'] = (nuc_props['bbox-5']+(sx-(nuc_props['bbox-5']-nuc_props['bbox-2']))//2).clip(lower=0, upper=nuc_image.shape[2])
        
        # Add intensity image cropped to larger bounding box
        #nuc_props['l_int_image'] = [nuc_image[slice(row['lbbox-0'],row['lbbox-3']),
        #                                     slice(row['lbbox-1'],row['lbbox-4']),
        #                                     slice(row['lbbox-2'],row['lbbox-5'])] for i, row in nuc_props.iterrows()]
    
    return nuc_labels, nuc_props

def nuc_segmentation_watershed(nuc_img, bg_thresh = 800, fg_thresh = 5000):

    from skimage.morphology import label
    from skimage.filters import sobel
    from skimage.segmentation import watershed

    edges = sobel(nuc_img)
    markers = np.zeros_like(nuc_img)
    foreground, background = 1, 2
    markers[nuc_img < bg_thresh] = background
    markers[nuc_img > fg_thresh] = foreground
    ws = watershed(edges, markers)
    seg = label(ws == foreground)
    return seg

def combine_overviews_ND2(input_folder, tidx = 0, align_channels=[1,1], ds = 2):
    from nd2reader import ND2Reader  
    '''[summary]

    Args:
        input_folder ([type]): Folder with ND2 overview images. Script assumes format is CYX (no Z-stacks)
        tidx (int, optional): Index of template image to use. Defaults to 0.
        align_channels (list, optional): Which channels to align the images. Length must match number of images. Defaults to [1,1].
        ds (int, optional): Downsampling. Defaults to 2.

    Returns:
        [type]: [description]
    '''
    paths = sorted(glob.glob(input_folder+os.sep+'*.nd2'))
    print(paths)
    imgs = []
    for path in paths:
        ND = ND2Reader(path)
        img = np.stack([ND.get_frame_2D(c=i) for i in range(ND.sizes['c'])]).astype(np.uint16)
        imgs.append(img)
        
    Y, X = imgs[tidx][align_channels[tidx]].shape
    ref_img = imgs[tidx][align_channels[tidx], Y*3//5:Y*4//5:ds, X*3//5:X*4//5:ds]

    imgs_reg = [imgs[tidx]]
    for i, off_img in enumerate(imgs):
        if i == tidx:
            continue
        off_img = pad_to_shape(off_img, imgs[tidx].shape)
        o_img = off_img[align_channels[i],Y*3//5:Y*4//5:ds, X*3//5:X*4//5:ds]
        shift, _, _ = phase_cross_correlation(ref_img, o_img) * ds
        print(shift)
        off_img_reg = ndi.shift(off_img, (0, shift[0], shift[1]), mode='constant', order=1).astype(np.uint16)
        imgs_reg.append(off_img_reg)

    imgs = np.concatenate(imgs_reg, axis=0)
    print(imgs.shape)

    return imgs, np.stack([ref_img, o_img])

def extract_cell_features(nuc_img, int_imgs: list, nuc_bg_int=800, nuc_fg_int=5000, scale=True):
    from sklearn import preprocessing

    nuc_masks = nuc_segmentation_watershed(nuc_img, bg_thresh = nuc_bg_int, fg_thresh = nuc_fg_int)
    expanded = expand_labels(nuc_masks, distance=5)
    props = [regionprops_table(expanded, int_img, properties=('mean_intensity',))['mean_intensity'] for int_img in int_imgs]
    res = np.vstack(props).T
    if scale:
        scaler = preprocessing.StandardScaler()
        scaler_model = scaler.fit(res)
        features = scaler_model.transform(res)
    else:
        features = res
        scaler_model = None
    return features, scaler_model

def relabel_nucs(nuc_image):
    from skimage.morphology import label
    out = mask_to_binary(nuc_image)
    out = label(out)
    return out.astype(nuc_image.dtype)

def full_frame_dc_to_single_nuc_dc(old_dc_path, new_dc_position_list, new_dc_path):
    new_drifts = []
    drifts = pd.read_csv(old_dc_path)
    for i, p in enumerate(new_dc_position_list):
        pos_name = p.split('_')[0]
        pos_drifts = drifts.query('position == @pos_name')
        for j, d in pos_drifts.iterrows():
            new_drifts.append(d.values.tolist()+[p])
    new_drifts = pd.DataFrame(new_drifts, columns = ['old_index','frame','z_px_course','y_px_course','x_px_course','z_px_fine',
    'y_px_fine','x_px_fine','orig_position', 'position'])
    new_drifts['z_px_course', 'y_px_course', 'x_px_course'] = 0
    new_drifts.to_csv(new_dc_path)
