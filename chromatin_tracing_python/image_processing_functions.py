# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:37:00 2020

@author: ellenberg
"""

import sys
import io
import yaml
import aicsimageio as aio
import czifile as cz
import os
import re
import numpy as np
import pandas as pd
import napari
from xml.etree import cElementTree as ElementTree
from skimage.filters import gaussian
from skimage.registration import phase_cross_correlation
from skimage.transform import resize
from scipy.stats import trim_mean
from skimage.measure import regionprops_table
import scipy.ndimage as ndi
import h5py
import dask
import dask.array as da
import itertools
import tifffile as tiff
from read_roi import read_roi_zip, read_roi_file
from flowdec import data as fd_data
from flowdec import restoration as fd_restoration
from flowdec import psf as fd_psf

def status_bar(n):
    for i in range(n):
        sys.stdout.write("\r[{:{}}] {:.1f}%".format("="*i, n-1, (100/(n-1)*i)))
        yield

def images_to_dask(folder, template):
    '''Wrapper function to generate dask arrays from image folder.

    Args:
        folder (string): path to folder
        template (list of strings): templates files in folder should match.

    Returns:
        dask array
        list of groups identified, currectly hardcoded to re_phrase='W[0-9]{4}'
    '''        
    print("Loading files to dask array: ")
    if '.h5' in template:
        x, groups = svih5_to_dask(folder, template)
    elif '.czi' in template or '.tif' in template or '.tiff' in template:
        x, groups = czi_lazy_to_dask(folder, template)
    print('\n Loaded images of shape: ', x.shape)
    print('Found positions ', groups)
    return x, groups


def svih5_to_dask(folder, template):
    all_files = all_matching_files_in_subfolders(folder, template)
    grouped_files, groups = group_filelist(all_files, re_phrase='W[0-9]{4}')
    progress = status_bar(len(all_files))
    pos_stack=[]
    with h5py.File(all_files[0], mode='r') as f:
        shape = f[list(f.keys())[0]]['ImageData']['Image'].shape

    for g in grouped_files:
        dask_arrays = []
        for fn in g:
            next(progress)
            f = h5py.File(fn, mode='r')
            d = f[list(f.keys())[0]]['ImageData']['Image']
            array = da.from_array(d, chunks=(1, 1, 1, shape[-2], shape[-1]))
            dask_arrays.append(array)
        pos_stack.append(da.stack(dask_arrays, axis=0))
    x = da.stack(pos_stack, axis=0)[...,0,:,:,:]
    print('Loaded images, final shape ', x.shape)
    return x, groups

def czi_tif_to_dask(folder, template):
    all_files = all_matching_files_in_subfolders(folder, template)
    grouped_files, groups = group_filelist(all_files, re_phrase='W[0-9]{4}')
    #print(groups, pos_list)
    
    if '.czi' in template:
        sample = read_czi_image(all_files[0])
    elif '.tif' in template or '.tiff' in template:
        sample = read_tif_image(all_files[0])
    else:
        raise TypeError('Input filetype not yet implemented.')

    pos_stack=[]
    for g in grouped_files:
        dask_arrays = []
        for fn in g:
            if '.czi' in template:
                d = dask.delayed(read_czi_image)(fn)
            elif '.tif' in template or '.tiff' in template:
                d = dask.delayed(read_tif_image)(fn)
            array = da.from_delayed(d, shape=sample.shape, dtype=sample.dtype)
            dask_arrays.append(array)
        pos_stack.append(da.stack(dask_arrays, axis=0))
    x = da.stack(pos_stack, axis=0)
    
    return x, groups

def czi_lazy_to_dask(folder, template):
    all_files = all_matching_files_in_subfolders(folder, template)
    grouped_files, groups = group_filelist(all_files, re_phrase='W[0-9]{4}')
    progress = status_bar(len(all_files))
    group_array=[]
    for g in grouped_files:
        pos_stack = []
        for fn in g:
            next(progress)
            img = aio.AICSImage(fn, chunk_by_dims=["Y", "X"])
            pos_stack.append(img.get_image_dask_data("CZYX", S=0, T=0, B=0, V=0))
        pos_stack = da.stack(pos_stack)
        group_array.append(pos_stack)
    x = da.stack(group_array)

    return x, groups
'''
### Does not works so well ###
def czi_lazy_to_dask_czifile(folder, template):
    all_files = all_matching_files_in_subfolders(folder, template)
    grouped_files, groups = group_filelist(all_files, re_phrase='W[0-9]{4}')
    progress = status_bar(len(all_files))
    group_array=[]
    sample = cz.CziFile(all_files[0])
    sample_shape = sample.subblock_directory[0].data_segment().data().shape
    sample_dtype = sample.dtype
    print('Loading images: ', sample_shape)
    for g in grouped_files:
        pos_stack = []
        for fn in g:
            next(progress)
            img = cz.CziFile(fn)
            single_stack = []
            for seg in img.subblock_directory:
                d = da.from_delayed(dask.delayed(seg.data_segment().data)(),
                shape=sample_shape, dtype=sample_dtype)
                single_stack.append(d[0,0,0,0,0,:,:,0])
            pos_stack.append(da.stack(single_stack))
        pos_stack = da.stack(pos_stack)
        group_array.append(pos_stack)
    x = da.stack(group_array)

    return x, groups
    '''
def rois_from_csv(path):
    rois = pd.read_csv(path)
    print('Loaded existing ROIs from ', path)
    return rois

def rois_from_imagej(roi_folder_path, template = '.zip', crop_size_z = 16, roi_scale = 0.5):
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

def roi_to_napari_shape(roi_table, position):
    rois_at_pos = roi_table[roi_table['position']==position]
    roi_shapes = []
    for i, roi in rois_at_pos.iterrows():
        roi_shape = np.array([[roi['y_min'], roi['x_min']],[roi['y_max'], roi['x_max']]])
        roi_shapes.append(roi_shape)
    roi_props = {'roi_id': rois_at_pos['roi_id'].values}
    return roi_shapes, roi_props

def roi_to_napari_points(roi_table, position):
    rois_at_pos = roi_table[roi_table['position']==position]
    roi_shapes = []
    for i, roi in rois_at_pos.iterrows():
        roi_shape = [roi['zc'], roi['yc'], roi['xc']]
        roi_shapes.append(roi_shape)
    roi_shapes = np.array(roi_shapes)
    roi_props = {'roi_id': rois_at_pos['index'].values}
    return roi_shapes, roi_props

def update_roi_shapes(shapes_layer, roi_table, position):
    rois = roi_table.copy()
    new_rois = rois[rois['roi_id'].isin(points_layer.properties['roi_id'])]
    rois[rois['position']==position] = new_rois
    rois = rois.dropna()
    return rois

def update_roi_points(point_layer, roi_table, position, downscale):
    rois = roi_table.copy()
    new_rois = pd.DataFrame(point_layer.data*downscale, columns=['zc','yc', 'xc'])
    new_rois.index.name = 'roi_id'
    new_rois = new_rois.reset_index()
    new_rois['position'] = position
    rois = rois.drop(rois[rois['position']==position].index)
    return pd.concat([rois, new_rois]).sort_values('position')

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
        image=czi.asarray()[0,0,:,0,:,::-1,:,0]
    return image


def read_tif_image(image_path):
    with tiff.TiffFile(image_path) as tif:
        image=tif.asarray()
    return image

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
                   
def all_matching_files_in_subfolders(path, template):
    '''
    Generates a sorted list of all files with the template in the 
    filename in directory and subdirectories.
    '''

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if all([s in file for s in template]):
                files.append(os.path.join(r, file))
    return sorted(files)

def group_filelist(input_list, re_phrase):
    '''
    Takes a list of strings (typically filepaths) and groups them according
    to a given element given by its position after splitting the string at split_char.
    E.g.for '..._WXXXX_PXXXX_TXXXX.ext' format this will by split_char='_' and element = -3.
    Returns a list of the , and 
    '''
    grouped_list = []
    groups=[]
    for k, g in itertools.groupby(sorted(input_list),
                                  lambda x: re.search(re_phrase, x).group(0)):
        grouped_list.append(list(g))
        groups.append(k)
    return grouped_list, groups

def match_file_lists(t_list,o_list):
    t_list_match = [item.split('__')[0][-5:] for item in t_list]
    o_list_match = [item for item in o_list if item.split('__')[0][-5:] in t_list_match]
    return o_list_match

def match_file_lists_decon(t_list,o_list):
    t_list_match = [item.split('\\')[-1].split('__')[0][-5:] for item in t_list]

    o_list_match = [item for item in o_list if 
                    item.split('\\')[-1].split('_P0001_')[0][-5:] in t_list_match]

    return o_list_match

def image_from_svih5(path,ch=None,index=(slice(None),
                                    slice(None),
                                    slice(None))):
    '''
    Parameters
    ----------
    path : String with file path to h5 file.
    index : Tuple with slice indexes of h5 file. Assumed CTZYX order.
            Default is all slices.
    
    Returns
    -------
    Image as numpy array.
    '''    
    with h5py.File(path, 'r') as f:
        if ch is not None:
            index=(slice(ch,ch+1),slice(None),)+index
            img=f[list(f.keys())[0]]['ImageData']['Image'][index][()][0,0]
        else:
            index=(slice(None),slice(None))+index
            img=f[list(f.keys())[0]]['ImageData']['Image'][index][()][:,0]
        
    return img

    
def detect_spots(img, spot_threshold):
    #Threshold, dilate and label image
    dog = gaussian(img, 1) - gaussian(img, 3)
    grad = np.sum(np.abs(np.gradient(img)), axis=0)
    img = img*dog*grad
    spot_img, num_spots = ndi.label(img>spot_threshold)
    
    #Make a DataFrame with the ROI info
    spot_props=pd.DataFrame(regionprops_table(spot_img, 
                                        properties=('label','centroid')))
    
    spot_props.drop(['label'], axis=1, inplace=True)
    spot_props.rename(columns={'centroid-0': 'zc',
                                        'centroid-1': 'yc',
                                        'centroid-2': 'xc',
                                        'index':'roi_id'},
                        inplace = True)
    print(f'Found {num_spots} spots.')
    return spot_props
    #Cleanup and saving of the DataFrame
        
def drift_corr_course(t_img, o_img, downsample=2):
    '''
    Calculates course and fine 
    drift between two svih5 images by phase cross correlation.

    Parameters
    ----------
    t_path : Path to template image in svih5 format.
    o_path : Path to offset image in svih5 format.
    ch : Which channel to use for drift correction.

    Returns
    -------
    A list of zyx course drifts and fine drifts (compared to course)

    '''        
    #Calculate course drift
    s = tuple(slice(None, None, downsample) for i in t_img.shape)
    course_drift=phase_cross_correlation(t_img[s], o_img[s], return_error=False) * downsample
    #Shift image for fine drift correction
    #o_img=ndi.shift(o_img,course_drift,order=0)
    return course_drift.tolist()

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

    #Label fiducial candidates and find maxima.
    t_img_label,num_labels=ndi.label(t_img>threshold)
    t_img_maxima=np.array(ndi.measurements.maximum_position(t_img, 
                                                labels=t_img_label, 
                                                index=range(num_labels)))
    
    #Filter maxima so not too close to edge and bright enough.
    t_img_maxima=np.array([m for m in t_img_maxima 
                            if min_bead_int<t_img[tuple(m)]
                            and all(m>8)])
    
    #Select random fiducial candidates. Seeded for reproducibility.
    np.random.seed(1)
    rand_points = t_img_maxima[np.random.choice(t_img_maxima.shape[0], size=n_points), :]
    
    #Initialize array to store shifts for all selected fiducials.
    shifts=np.empty_like(rand_points, dtype=np.float32)
    
    #Calculate fine scale drift for all selected fiducials.
    
    for i, point in enumerate(rand_points):
        print(point, course_drift)
        s_t = tuple([slice(ind-8, ind+8) for ind in point])
        s_o = tuple([slice(ind-int(shift)-8, ind-int(shift)+8) for (ind, shift) in zip(point, course_drift)])
        try:
            shift = phase_cross_correlation(t_img[s_t], 
                                        o_img[s_o], 
                                        upsample_factor=upsampling,
                                        return_error=False)
        except ValueError: #In case point is too close to edge of image.
            shifts[i] = [0 for p in point]
        else:
            shifts[i] = shift
        
    #Return the 60% central mean to avoid outliers.
    return trim_mean(shifts, proportiontocut=0.2, axis=0)#, np.std(shifts, axis=0)

def napari_view(img, flat = True, points=None, downscale=2, trace_ch=0, ref_slice=0):
    with napari.gui_qt():
        viewer = napari.view_image(img[...,::downscale,::downscale,::downscale], contrast_limits=(0,2000))
        if points is not None:
            point_layer = viewer.add_points(points/downscale, 
                                                    size=8,
                                                    edge_width=3,
                                                    edge_color='red',
                                                    face_color='transparent',
                                                    n_dimensional=True)
            sel_dim = [ref_slice, trace_ch] + list(points[0,:]/downscale)
            for dim in range(len(sel_dim)):
                viewer.dims.set_current_step(dim, sel_dim[dim])

    if points is not None:
        return point_layer

def decon_RL_setup():
    algo = fd_restoration.RichardsonLucyDeconvolver(3).initialize()
    kernel = fd_psf.GibsonLanni(
            size_x=16, size_y=16, size_z=16, pz=0., wavelength=.610,
            na=1.46, res_lateral=.1, res_axial=.15
        ).generate()
    return algo, kernel

def decon_RL(img, kernel, algo, niter=30):
    res = algo.run(fd_data.Acquisition(data=img, kernel=kernel), niter=niter).data
    return res