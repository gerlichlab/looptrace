# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 06:23:05 2020

@author: ellenberg
"""

import numpy as np
from skimage.feature import ORB
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import pdist
from skimage.filters import threshold_otsu, gaussian

def comp_ssim(img1, img2):
    '''
    Calculates structrual similarity index
    '''
    #shift_int=np.abs((shift/2).astype(np.int))
    #s=[slice(shift_int[0],-shift_int[0]),slice(shift_int[1],-shift_int[1]),slice(shift_int[2],-shift_int[2])]       
    #ssim_out, ssim_image = ssim(img1[s[0],s[1],s[2]],shifted_img[s[0],s[1],s[2]],full=True)
    #if thresh:
    #    thresh = threshold_otsu(img1)
    #    img1 = img1*(img1>thresh)
    #    img2 = img2*(img2>thresh)
    ssim_out, ssim_image = ssim(img1,img2, full=True,  gaussian_weights=True)
    #if plot:
    #    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
    #    ax1.imshow(img1[15],cmap='gray')
    #    ax2.imshow(img2[15],cmap='gray')
    #    ax3.imshow(ssim_image[15],cmap='gist_heat')

    return ssim_out, ssim_image
    
def comp_pcc_man_coloc(img1,img2):
    
    #if thresh==True:
    #    thresh = threshold_otsu(img1)
    #    img1 = img1*(img1>thresh)
    #    img2 = img2*(img2>thresh)
    img1=img1.astype(np.float32)
    img2=img2.astype(np.float32)
    img1_bar=np.mean(img1)
    img2_bar=np.mean(img2)
    
    pcc_num=np.sum((img1-img1_bar)*(img2-img2_bar))
    pcc_denom=np.sqrt(np.sum((img1-img1_bar)**2)*np.sum((img2-img2_bar)**2))
    pcc=pcc_num/pcc_denom
    
    mac=np.sum(img1*img2)/np.sqrt(np.sum(img1**2)*np.sum(img2**2))

    return pcc,mac   
    
def comp_orb_ratio(img1, img2):
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
    img1=img1>0
    img2=img2>0
    area_ratio = np.sum(img2)/np.sum(img1)
    iou = np.sum(img1*img2)/np.sum(img1+img2)
    
    return area_ratio, iou