# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:37:00 2020

@author: ellenberg
"""

import io
import yaml
import czifile as cz
import tifffile as tiff
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from xml.etree import cElementTree as ElementTree
from skimage.segmentation import clear_border
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import regionprops_table
from skimage.morphology import dilation, square
from skimage.feature import register_translation, ORB, match_descriptors, plot_matches
from skimage.metrics import structural_similarity as ssim
from skimage.exposure import match_histograms, rescale_intensity
from skimage.restoration import richardson_lucy, wiener
from scipy.spatial.distance import pdist
from scipy import ndimage as ndi

from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.metrics import pairwise_distances
import hdbscan
import microscPSF.microscPSF as msPSF
from flowdec import data as fd_data
from flowdec import restoration as fd_restoration
from sklearn.neighbors import KernelDensity


def load_config(config_file):
    '''
    Open config file and return config variable form yaml file.
    '''
    with open(config_file, 'r') as stream:
        try:
            config=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def read_czi_image(image_path):
    '''
    Reads czi files as arrays using czifile package. Returns only CZYX image.
    '''
    with cz.CziFile(image_path) as czi:
        image=czi.asarray()[0,0,:,0,:,:,:,0]
    return image


def read_tif_image(image_path):
    with tiff.TiffFile(image_path) as tif:
        image=tif.asarray()
    return image

def psf_gen_GL():
    mp = msPSF.m_params
    mp['M']=63
    mp['NA']=1.46
    pixel_size = 0.1
    rv = np.arange(0.0, 3.01, pixel_size)
    zv = np.arange(-1.5, 1.51, pixel_size*2)
    psf_xyz = msPSF.gLXYZFocalScan(mp, pixel_size, 31, zv, pz = 0.0)
    psfSlicePics(psf_xyz, 15, 7, zv)
    return psf_xyz

def psfSlicePics(psf, sxy, sz, zvals, pixel_size = 0.1):
    ex = pixel_size * 0.5 * psf.shape[1]
    fig = plt.figure(figsize = (12,4))
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(np.sqrt(psf[sz,:,:]),
               interpolation = 'none', 
               extent = [-ex, ex, -ex, ex],
               cmap = "gray")
    ax1.set_title("PSF XY slice")
    ax1.set_xlabel(r'x, $\mu m$')
    ax1.set_ylabel(r'y, $\mu m$')

    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(np.sqrt(psf[:,:,sxy]),
               interpolation = 'none',
               extent = [-ex, ex, zvals.max(), zvals.min()],
               cmap = "gray")
    ax2.set_title("PSF YZ slice")
    ax2.set_xlabel(r'y, $\mu m$')
    ax2.set_ylabel(r'z, $\mu m$')

    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow(np.sqrt(psf[:,sxy,:]), 
               interpolation = 'none',
               extent = [-ex, ex, zvals.max(), zvals.min()],
               cmap = "gray")
    ax3.set_title("PSF XZ slice")
    ax3.set_xlabel(r'x, $\mu m$')
    ax3.set_ylabel(r'z, $\mu m$')

    plt.show()
    
def RL_decon(data, kernel, iterations):
    algo = fd_restoration.RichardsonLucyDeconvolver(data.ndim).initialize()
    res = algo.run(fd_data.Acquisition(data=data, kernel=kernel), niter=iterations).data
    return res

def merge_multipos_lsm(folder):
    image_paths=all_matching_files_in_subfolders(folder,['.lsm'])
    images=[tiff.imread(img) for img in image_paths]
    images=np.concatenate(images, axis=1)
    for pos in range(images.shape[1]):
        tiff.imsave(folder+'_Pos000'+str(pos)+'.tiff',images[pos],imagej=True)
        
def drift_corr_timelapse_tiff(folder, nuc_ch, output_folder):
    image_paths=all_matching_files_in_subfolders(folder,['.tif'])
    for path in image_paths:
        img=tiff.imread(path)
        print('Loaded ', path)
        output = drift_shift_timelapse(img, nuc_ch)
        filename=path.split('\\')[-1].split('.')[0]
        tiff.imsave(output_folder+os.sep+filename+'_DC.tiff',output,imagej=True)
        print('Saved ', output_folder+os.sep+filename+'_DC.tiff')
        
def drift_shift_timelapse(t_image, ch):
    output=np.zeros_like(t_image)
    
    template_image=t_image[0,:,ch,:,:]
    
    def _drift_shift_pl(template_img, offset_img, ch):
        t_img=ndi.sobel(gaussian(template_img,1))
        o_img=ndi.sobel(gaussian(offset_img[:,ch,:,:],1))
        shift=drift_corr_cc(t_img, o_img, upsampling=2, downsampling=1)
        shift=np.insert(shift,1,0)
        shifted_img=ndi.shift(offset_img, shift)
        return shifted_img
    output=Parallel(n_jobs=-1)(delayed(_drift_shift_pl)(template_image,t_image[i],ch) for i in range(1,t_image.shape[0]))
    output=[t_image[0]]+output
    output=np.stack(output,axis=0)
    return output

def read_czi_meta(image_path, tags, save_meta=False):
    '''
    Function to read metadata and image data for CZI files.
    Define the information to be extracted from the xml tags dict in config file.
    Optionally a YAML file with the metadata can be saved in the same path as the image.
    Return a dictionary with the extracted metadata.
    '''
    def parser(data, tags):
        tree = ElementTree.iterparse(data, events=('start',))
        _, root = next(tree)
    
        for event, node in tree:
            if node.tag in tags:
                yield node.tag, node.text
            root.clear()
    
    with cz.CziFile(image_path) as czi:
        meta=czi.metadata()
    
    with io.StringIO(meta) as f:
        results = parser(f, tags)
        metadict={} 
        for tag, text in results:
            metadict[tag]=text
    if save_meta:
        with open(image_path[:-4]+'_meta.yaml', 'w') as myfile:
            yaml.safe_dump(metadict, myfile)
    return metadict

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
    
def pad_to_shape(arr, shape):
    '''
    Pads an array with zeros to a given shape (list or tuple).
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
        arr=pad_to_shape(arr, exp_shape)
        arr=crop_to_shape(arr,shape)
        return arr
    ps = tuple((n,n) for n in p)
    arr=np.pad(arr,ps)
    if arr.shape == shape:
        return arr
    else:
        p=np.subtract(shape,arr.shape)
        ps = tuple((0,n) for n in p)
        arr=np.pad(arr,ps)
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
    s=tuple([slice(pos,pos+si) for pos,si in zip(tl_pos,size)])
    return arr[s]
        
def drift_corr_cc(t_img, o_img, upsampling=1, downsampling=1):
    '''
    Performs drift correction by cross-correlation.
    Image can be upsampled for sub-pixel accuracy,
    or downsampled (by simple slicing) for speed based on reduced sampling.
    Works in 2D or 3D (ZYX), if 3D with channels (CZYX) then averages shift found for each channel.
    '''
    s = slice(None,None,downsampling)
    if t_img.ndim == 2:
        shift = register_translation(t_img[s,s], o_img[s,s], upsample_factor=upsampling, return_error=False)
        shift = shift * [downsampling, downsampling] 
        return shift
    elif t_img.ndim == 3:
        shift = register_translation(t_img[:,s,s], o_img[:,s,s], upsample_factor=upsampling, return_error=False)
        shift = shift * [1, downsampling, downsampling] 
        return shift
    elif t_img.ndim == 4:
        shift_all=[]
        for ch in range(t_img.shape[0]):
            shift = register_translation(t_img[ch,:,s,s], o_img[ch,:,s,s], upsample_factor=upsampling, return_error=False)
            shift_all.append(shift * [1, downsampling, downsampling])
        shift = np.mean(np.array(shift_all),axis=0)
        return shift
    else:
        raise TypeError('Unsupported number of dimensions.')
            
def drift_corr_fine(template_image, offset_image, upsampling=100):
    '''
    Performs 3d drift correction of one identified bead by cross-correlation.
    Image can be upsampled for sub-pixel accuracy.
    TODO: Increase number of beads/points used by implementing a peak finding algorithm.
    TODO: Adapt to multichannel images.
    '''
    max_proj=np.max(template_image,axis=0)
    max_ind=np.unravel_index(np.argmax(max_proj, axis=None), max_proj.shape)
    s=[slice(ind-16, ind+16) for ind in max_ind]
    shift = register_translation(template_image[:,s[0],s[1]], offset_image[:,s[0],s[1]], upsample_factor=upsampling, return_error=False)
    return shift
    
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
                   
def all_matching_files_in_subfolders(path, template):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if all([s in file for s in template]):
                files.append(os.path.join(r, file))
    return files

def match_file_lists(t_list,o_list,match_idx):
    t_list_match = [item[match_idx[0]:match_idx[1]] for item in t_list]
    o_list_match = [item for item in o_list if item[match_idx[2]:match_idx[3]] in t_list_match]
    return o_list_match

def render_gauss_const(df,sigma,cam_px,cam_nm,grid_px_size, output_res):
    X = df[['xnm','ynm']].to_numpy()/grid_px_size
    X=np.round(X).astype(int)
    grid=np.zeros((cam_px*cam_nm//grid_px_size,cam_px*cam_nm//grid_px_size))
    point=np.zeros((13,13))
    point[6,6]=1
    point=ndi.gaussian_filter(point,sigma)

    for xval,yval in zip(X[:,0],X[:,1]):
        grid[xval-6:xval+7,yval-6:yval+7]+=point
        
    grid=cv2.resize(grid,(output_res, output_res))
    grid=np.flip(np.rot90(grid),0)
    plt.imshow(np.clip(grid,0,1),cmap='gist_heat')
    return grid

def render_gauss(df,pixel_size):
    X = df[['xnm','ynm']].to_numpy()/pixel_size
    X=np.round(X).astype(int)
    err = df['xnmerr'].to_numpy()/pixel_size
    grid=np.zeros((512*103//pixel_size,512*103//pixel_size))
    point=np.zeros((13,13))
    point[6,6]=1

    for xval,yval,e in zip(X[:,0],X[:,1],err):
        point=ndi.gaussian_filter(point,e)
        grid[xval-6:xval+7,yval-6:yval+7]+=point
        
    grid=cv2.resize(grid,(512,512))
    plt.imshow(np.flip(np.rot90(np.clip(grid,0,0.5)),0),cmap='gist_heat')

def render_hist(df):
    X = df[['xnm','ynm']].to_numpy()/pixel_size
    heatmap, xedges, yedges = np.histogram2d(X[:,0], X[:,1], bins=1000)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    heatmap = cv2.resize(np.flip(heatmap.T,(0)),(512,512))
    plt.imshow(np.clip(heatmap,0,10), extent=extent, origin='lower',cmap='gist_heat')
    
    
def clust_dbscan(df, eps, min_samples, hdb=False):
    X = df[['xnm','ynm']].to_numpy()
    xi, yi = np.mgrid[0:np.max(X[:,0]):512j,0:np.max(X[:,1]):512j]
    # #############################################################################
    # Compute DBSCAN
    print('Total number of localizations: %d' % X.shape[0])
    if hdb:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_samples)
        labels = clusterer.fit_predict(X)
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[labels>-1] = True
        
    else:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    
    
    # #############################################################################
    '''
    Plotting:
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = (labels == k)
    
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=None, markersize=5)
    
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], '.',
                 markeredgecolor='k', markersize=0.05)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    '''
    return core_samples_mask
    
def clust_kde(df):
    X = df[['xnm','ynm']].to_numpy()
    xi, yi = np.mgrid[0:np.max(X[:,0]):512j,0:np.max(X[:,1]):512j]
    xy_i=np.vstack((np.ravel(xi),np.ravel(yi))).T
    kde = KernelDensity(kernel='gaussian', bandwidth=10).fit(X)
    Z = np.exp(kde.score_samples(xy_i))
    plt.contourf(xi,yi,np.clip(Z.reshape(512,512),0,1e-9))
    
    
