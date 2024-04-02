# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from typing import *
import numpy as np
import numpy.typing as npt
import pandas as pd

import scipy.ndimage as ndi
from scipy.stats import trim_mean
from skimage.segmentation import clear_border, expand_labels
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import ball, remove_small_objects, white_tophat
from skimage.measure import regionprops_table

from looptrace.numeric_types import NumberLike
from looptrace.wrappers import phase_xcor

__author__ = "Kai Sandvold Beckwith"
__credits__ = ["Kai Sandvold Beckwith", "Vince Reuter"]

X_CENTER_COLNAME = "xc"
Y_CENTER_COLNAME = "yc"
Z_CENTER_COLNAME = "zc"
CENTROID_COLUMNS_REMAPPING = {
    "centroid_weighted-0": Z_CENTER_COLNAME, 
    "centroid_weighted-1": Y_CENTER_COLNAME, 
    "centroid_weighted-2": X_CENTER_COLNAME,
    }

PixelValue = Union[np.uint8, np.uint16]


def extract_labeled_centroids(img: npt.NDArray[PixelValue]) -> pd.DataFrame:
    props = regionprops_table(img, properties=("label", "centroid"))
    table = pd.DataFrame(props)
    return table.rename(columns=CENTROID_COLUMNS_REMAPPING)


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


def subtract_crosstalk(source, bleed, threshold=500):
    shift = drift_corr_coarse(source, bleed, downsample=1)
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


def detect_spots(input_img, spot_threshold: NumberLike, expand_px: int = 10):
    """Spot detection by difference of Gaussians filter

    Arguments
    ---------
    img : ndarray
        Input 3D image
    spot_threshold : NumberLike
        Threshold to use for spots
    expand_px : int
        Number of pixels by which to expand contiguous subregion, 
        up to point of overlap with neighboring subregion of image
        
    Returns
    -------
    pd.DataFrame, np.ndarray, np.ndarray: 
        The centroids and roi_IDs of the spots found, 
        the image used for spot detection, and 
        numpy array with only sufficiently large regions 
        retained (bigger than threshold number of pixels), 
        and dilated by expansion amount (possibly)
    """
    # TODO: Do not use hard-coded sigma values (second parameter to gaussian(...)).
    # See: https://github.com/gerlichlab/looptrace/issues/124

    img = white_tophat(image=input_img, footprint=ball(2))
    img = gaussian(img, 0.8) - gaussian(img, 1.3)
    img = img / gaussian(input_img, 3)
    img = (img - np.mean(img)) / np.std(img)
    labels, _ = ndi.label(img > spot_threshold)
    labels = expand_labels(labels, expand_px)
    
    # Make a DataFrame with the ROI info.
    spot_props = _reindex_to_roi_id(pd.DataFrame(regionprops_table(
        label_image=labels, 
        intensity_image=input_img, 
        properties=('label', 'centroid_weighted', 'intensity_mean')
        )).drop(['label'], axis=1).rename(columns=CENTROID_COLUMNS_REMAPPING))

    return spot_props, img, labels


def detect_spots_int(input_img, spot_threshold: NumberLike, expand_px: int = 1):
    """Spot detection by intensity filter

    Arguments
    ---------
    img : ndarray
        Input 3D image
    spot_threshold : NumberLike
        Threshold to use for spots
    expand_px : int
        Number of pixels by which to expand contiguous subregion, 
        up to point of overlap with neighboring subregion of image
        
    Returns
    -------
    pd.DataFrame, np.ndarray, np.ndarray: 
        The centroids and roi_IDs of the spots found, 
        the image used for spot detection, and 
        numpy array with only sufficiently large regions 
        retained (bigger than threshold number of pixels), 
        and dilated by expansion amount (possibly)
    """
    # TODO: enforce that output column names don't vary with code path walked.
    # See: https://github.com/gerlichlab/looptrace/issues/125

    binary = input_img > spot_threshold
    binary = ndi.binary_fill_holes(binary)
    struct = ndi.generate_binary_structure(input_img.ndim, 2)
    labels, n_obj = ndi.label(binary, structure=struct)
    if n_obj > 1: # Do not need this with area filtering below
        labels = remove_small_objects(labels, min_size=5)
    if expand_px > 0:
        labels = expand_labels(labels, expand_px)
    if np.all(labels == 0): # No substructures (ROIs) exist after filtering.
        spot_props = pd.DataFrame(columns=['label', 'z_min', 'y_min', 'x_min', 'z_max', 'y_max', 'x_max', 'area', 'zc', 'yc', 'xc', 'intensity_mean'])
    else:
        spot_props = _reindex_to_roi_id(
            pd.DataFrame(regionprops_table(
                labels, input_img, properties=('label', 'bbox', 'area', 'centroid_weighted', 'intensity_mean')
                )).rename(
                    columns={**CENTROID_COLUMNS_REMAPPING,
                        'bbox-0': 'z_min',
                        'bbox-1': 'y_min',
                        'bbox-2': 'x_min',
                        'bbox-3': 'z_max',
                        'bbox-4': 'y_max',
                        'bbox-5': 'x_max'
                        }
                    )
            )

    return spot_props, input_img, labels


def drift_corr_coarse(t_img, o_img, downsample=1):
    '''
    Calculates coarse and fine 
    drift between two svih5 images by phase cross correlation.

    Parameters
    ----------
    t_path : Path to template image in svih5 format.
    o_path : Path to offset image in svih5 format.

    Returns
    -------
    A list of zyx coarse drifts and fine drifts (compared to coarse)

    '''        
    s = tuple(slice(None, None, downsample) for i in t_img.shape)
    coarse_drift = phase_xcor(np.array(t_img[s]), np.array(o_img[s])) * downsample
    return coarse_drift

def drift_corr_multipoint_cc(t_img, o_img, coarse_drift, threshold, min_bead_int, n_points=50, upsampling=100):
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
        s_o = tuple([slice(ind-int(shift)-8, ind-int(shift)+8) for (ind, shift) in zip(point, coarse_drift)])
        t = t_img[s_t]
        o = o_img[s_o]
        
        try:
            shift = phase_xcor(t, o, upsample_factor=upsampling)
        except (ValueError, AttributeError):
            shift = np.array([0,0,0])
        return shift

    shifts = joblib.Parallel(n_jobs=-1, prefer='threads')(joblib.delayed(extract_and_correlate)(point, t_img, o_img, upsampling) for point in t_img_maxima)
    shifts = np.array(shifts)

    fine_drift = trim_mean(shifts, proportiontocut=0.2, axis=0)
    print(f'Fine drift: {fine_drift} with untrimmed STD of {np.std(shifts, axis=0)} .')
    #Return the 60% central mean to avoid outliers.
    return fine_drift#, np.std(shifts, axis=0)


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


def full_frame_dc_to_single_nuc_dc(old_dc_path, new_dc_position_list, new_dc_path):
    new_drifts = []
    drifts = pd.read_csv(old_dc_path)
    for i, p in enumerate(new_dc_position_list):
        pos_name = p.split('_')[0]
        pos_drifts = drifts.query('position == @pos_name')
        for j, d in pos_drifts.iterrows():
            new_drifts.append(d.values.tolist()+[p])
    new_drifts = pd.DataFrame(new_drifts, columns = ['old_index','frame','z_px_coarse','y_px_coarse','x_px_coarse','z_px_fine',
    'y_px_fine','x_px_fine','orig_position', 'position'])
    new_drifts['z_px_coarse', 'y_px_coarse', 'x_px_coarse'] = 0
    new_drifts.to_csv(new_dc_path)


def _index_as_roi_id(props_table: pd.DataFrame) -> pd.DataFrame:
    return props_table.rename(columns={'index': 'roi_id'})


def _reindex_to_roi_id(props_table: pd.DataFrame) -> pd.DataFrame:
    return _index_as_roi_id(props_table.reset_index(drop=True))
