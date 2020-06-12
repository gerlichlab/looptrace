# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 06:22:09 2020

@author: ellenberg
"""

def detect_nuclei(nuc_image, min_size, exp_bb=-1, clear_edges=True):
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