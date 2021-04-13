# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import numpy as np
from skimage.feature import ORB, local_binary_pattern
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import pdist
from scipy.stats import skew, kurtosis
from skimage.filters import threshold_otsu, gaussian
from numba import njit

def kullback_leibler_divergence(p, q):
    '''
    Used to score histogram differences, 0 is identical.
    
    Parameters
    ------------
    p, q: Any numerical list of corresponding values, typically histograms.

    Returns
    ------------
    The KLD score.

    '''

    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

def comp_lbp(img1, img2):
    '''
    Comparison of the central 2D slice of a 3D image stack by local binary patterns.
    Uses Kullback Leibler Divergence to score the similarity of the LBP histograms.
    '''

    if len(img1.shape) == 3:
        max_z = np.argmax(np.sum(img1, axis=((1,2))))
        lbp1 = local_binary_pattern(img1[max_z], 16, 2)
        lbp2 = local_binary_pattern(img2[max_z], 16, 2)
    else:
        lbp1 = local_binary_pattern(img1, 16, 2)
        lbp2 = local_binary_pattern(img2, 16, 2)
    
    n_bins = 100
    hist1, _ = np.histogram(lbp1, density=True, bins=n_bins, range=(0, n_bins))
    hist2, _ = np.histogram(lbp2, density=True, bins=n_bins, range=(0, n_bins))
    score = kullback_leibler_divergence(hist1, hist2)
    return score

def comp_var(img1, img2):
    '''
    Ratio of variances of two images.
    '''

    var1 = np.var(img1, axis=None)
    var2 = np.var(img2, axis=None)
    return var2/var1

def comp_skew(img1, img2):
    '''
    Ratios of the skews of two images.
    '''

    s1 = skew(img1, axis=None)
    s2 = skew(img2, axis=None)
    return s2/s1

def comp_kurtosis(img1, img2):
    '''
    Ratio of the kurtosis of the images.
    '''

    k1 = kurtosis(img1, axis=None)
    k2 = kurtosis(img2, axis=None)
    return k2/k1

def comp_ssim(img1, img2):
    '''
    Calculates structrual similarity index of two images.
    '''

    ssim_out = ssim(img1,img2, full=False,  win_size=5)

    return ssim_out

@njit    
def comp_pcc_coloc(img1,img2):
    '''
    Pixel-by-pixel correlation of two images by Pearson's correlation coefficient.
    '''

    img1=img1.astype(np.float32)
    img2=img2.astype(np.float32)
    img1_bar=np.mean(img1)
    img2_bar=np.mean(img2)
    
    pcc_num=np.sum((img1-img1_bar)*(img2-img2_bar))
    pcc_denom=np.sqrt(np.sum((img1-img1_bar)**2)*np.sum((img2-img2_bar)**2))
    pcc=pcc_num/pcc_denom
    
    return pcc 

@njit
def comp_mac_coloc(img1, img2):
    '''
    Pixel-by-pixel correlation of two images by Mander's correlation coefficient.
    '''
    mac=np.sum(img1*img2)/np.sqrt(np.sum(img1**2)*np.sum(img2**2))
    return mac
    
def comp_orb_ratio(img1, img2):
    '''
    Detects ORB keypoint features of two images, and calculates distance between all features.
    Returns the ratio of the median distance as a metric of image distortion.
    '''

    if img2.ndim > 2:
        z_max=np.argmax(np.sum(img1,axis=(1,2)))
        img1=img1[z_max,:,:]
        img2=img2[z_max,:,:]
                
    try:
        detector1 = ORB(n_keypoints=200)
        detector2 = ORB(n_keypoints=200)
        detector1.detect_and_extract(img1)
        detector2.detect_and_extract(img2)
        #matches = match_descriptors(detector1.descriptors, 
        #                           detector2.descriptors, 
        #                            cross_check=True)
        d1 = pdist(detector1.keypoints)
        d2 = pdist(detector2.keypoints)
        ratio = np.median(d2)/np.median(d1)
    except RuntimeError:
        ratio=np.nan
    return ratio
    
def comp_area_iou(img1, img2):
    '''
    Calculates the ratio of integrated pixel area and  
    intersection over union of area in two images over an Otsu threshold.
    '''

    img1=gaussian(img1,sigma=3)
    img2=gaussian(img2,sigma=3)
    try:
        thresh1=threshold_otsu(img1)
        thresh2=threshold_otsu(img2)
    except ValueError:
        #If for some reason whole image is 0 otsu threshold crashes.
        area_ratio=np.nan
        iou=np.nan
        return area_ratio, iou
    img1_thresh=img1>thresh1
    
    img2_thresh=img2>thresh2
    area_ratio = np.sum(img2_thresh)/np.sum(img1_thresh)
    iou = np.sum(img1_thresh*img2_thresh)/np.sum(img1_thresh+img2_thresh)
    return area_ratio, iou
    
def comp_area_iou_sr(img1,img2):
    '''
    Calculates the ratio of integrated pixel intensity and intersection over union
    for images with 0 background.
    '''
    
    img1=img1>0
    img2=img2>0
    area_ratio = np.sum(img2)/np.sum(img1)
    iou = np.sum(img1*img2)/np.sum(img1+img2)
    
    return area_ratio, iou