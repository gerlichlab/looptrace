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
from skimage.segmentation import clear_border, expand_labels
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import regionprops_table

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


# TODO: integrate numpydoc_decorator.
# @doc(
#     summary="Extract labeled (by index / integer) centroids from regions in the given image.",
#     parameters=dict(img="The image in which to find regional centroids"),
#     returns="Table of labels and centroid coordinates",
#     raises=dict(ValueError="If the given image is neither 2- nor 3-D.")
#     see_also=dict(regionprops_table="Function from scikit-image which computes the centroids"),
# )
def extract_labeled_centroids(img: npt.NDArray[PixelValue]) -> pd.DataFrame:
    centroid_key = "centroid"
    ndim = len(img.shape)
    # Here we need to account for 2D or 3D image, and the fact that these centroids can't be weighted, 
    # since we don't pass the intensity image, just rather the masks image.
    if ndim == 2:
        newcols = [Y_CENTER_COLNAME, X_CENTER_COLNAME]
    elif ndim == 3:
        newcols = [Z_CENTER_COLNAME, Y_CENTER_COLNAME, X_CENTER_COLNAME]
    else:
        raise ValueError("Image from which to extract region centroids is neither 2- nor 3-D, but {ndim}-D!")
    colname_mapping = {f"{centroid_key}-{i}": newcol for i, newcol in enumerate(newcols)}
    props = regionprops_table(img, properties=("label", centroid_key))
    table = pd.DataFrame(props)
    return table.rename(columns=colname_mapping, errors="raise", inplace=False)


def subtract_crosstalk(source, bleed, threshold=500):
    shift = drift_corr_coarse(source, bleed, downsample=1)
    bleed = ndi.shift(bleed, shift=shift, order=1)
    mask = bleed > threshold
    ratio = np.average(source[mask] / bleed[mask])
    print(ratio)
    out = np.clip(source - (ratio * bleed), a_min=0, a_max=None)
    return out, bleed


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
