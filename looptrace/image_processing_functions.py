# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import io
import yaml
import czifile
import os
import re
import numpy as np
import pandas as pd
import napari
from skimage.segmentation import clear_border
from skimage.filters import gaussian, threshold_otsu
from skimage.registration import phase_cross_correlation
from scipy.stats import trim_mean
from skimage.measure import regionprops_table
import scipy.ndimage as ndi
import dask
import dask.array as da
import zarr
import itertools
import tifffile
import joblib

def czi_to_tif(in_folder, template, out_folder, prefix):
    '''Convert CZI files from MyPIC experiment to single YX tif images.

    Args:
        in_folder (str): Top level folder path to find czi files
        template (list): Template to match files to
        out_folder (str): Output folder to save tif images
        prefix (str): Prefix of output files, prepended to axis info
    '''

    all_files = all_matching_files_in_subfolders(in_folder, template)
    sample = czifile.CziFile(all_files[0])
    n_c = sample.shape[-6]
    n_z = sample.shape[-4]

    def save_single_tif(n_c, n_z, path, out_folder):
        pos = re.search('W[0-9]{4}',path)[0]
        pos = 'P'+pos[1:]
        t = re.search('T[0-9]{4}',path)[0]
        img = czifile.imread(path)[0,0,:,0,:,:,:,0] 
        for c in range(n_c):
            for z in range(n_z):
                fn = out_folder + prefix + pos + '_' + t + '_C' + str(c).zfill(4)+ '_Z' + str(z).zfill(4)+'.tif'
                if not os.path.isfile(fn):
                    tifffile.imwrite(fn, img[c,z], compression='deflate', metadata={'axes': 'YX'})

    joblib.Parallel(n_jobs=-2)(joblib.delayed(save_single_tif)(n_c, n_z, path, out_folder) for path in all_files)

def tif_store_to_dask(folder, re_search = 'P[0-9]{4}'):
    '''Read a series of tif files as a zarr array from a single folder using tifffile sequence reader, 
    then assemble the sequences to a dask array.

    Args:
        folder (str): Path to folder with tif files
        prefix (str): Prefix of tif files in folder before axes info

    Returns:
        Dask array: Dask array with all the matching tif files form the folder
    '''
    imgs = []
    all_files = all_matching_files_in_subfolders(folder, ['.tif'])
    groups, positions = group_filelist(all_files, re_search)
    for i, group in enumerate(groups):
        print('Loading images for position ', positions[i])
        seq = tifffile.TiffSequence(group, pattern='axes')
        with seq.aszarr() as store:
            imgs.append(da.from_array(zarr.open(store, mode='r'), chunks =  (1,1,1,1,-1,-1))[0])
    return imgs
        

def images_to_dask(folder, template):
    '''Wrapper function to generate dask arrays from image folder.

    Args:
        folder (string): path to folder
        template (list of strings): templates files in folder should match.

    Returns:
        x: dask array
        groups : list of groups identified, currectly hardcoded to re_phrase='W[0-9]{4}'
    '''        
    print("Loading files to dask array: ")
    #if '.h5' in template:
    #    x, groups = svih5_to_dask(folder, template)
    if '.czi' in template or '.tif' in template or '.tiff' in template:
        x, groups, all_files = czi_tif_to_dask(folder, template)
    print('\n Loaded images of shape: ', x[0])
    print('Found positions ', groups)
    return x, groups, all_files

def czi_tif_to_dask(folder, template):
    ''' Read a series of tif or czi files into a virtual Dask array.
    Args:
        folder (string): path to folder with files (also in subfolders)
        template (list of strings): templates files in folder should match.

    Returns:
        pos_stack (list): list of dask arrays, one per position
        groups (list): list of groups identified, currectly hardcoded to re_phrase='W[0-9]{4}'
        all_files (list): list of all file paths read
    '''

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
    #x = da.stack(pos_stack, axis=0)
    
    return pos_stack, groups, all_files

'''

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

def aio_lazy_to_dask(folder, template):
    all_files = all_matching_files_in_subfolders(folder, template)
    grouped_files, groups = group_filelist(all_files, re_phrase='W[0-9]{4}')
    progress = status_bar(len(all_files))
    group_array=[]
    for g in grouped_files:
        pos_stack = []
        for fn in g:
            next(progress)
            img = aio.AICSImage(fn, chunk_by_dims=["Z", "Y", "X"])
            pos_stack.append(img.dask_data[0,0])
        pos_stack = da.stack(pos_stack)
        group_array.append(pos_stack)
    x = da.stack(group_array)

    return x, groups

### Does not works so well ###
def czi_lazy_to_dask_czifile(folder, template):
    all_files = all_matching_files_in_subfolders(folder, template)
    grouped_files, groups = group_filelist(all_files, re_phrase='W[0-9]{4}')
    progress = status_bar(len(all_files))
    group_array=[]
    sample = czifile.CziFile(all_files[0])
    sample_shape = sample.subblock_directory[0].data_segment().data().shape
    sample_dtype = sample.dtype
    print('Loading images: ', sample_shape)
    for g in grouped_files:
        pos_stack = []
        for fn in g:
            next(progress)
            img = czifile.CziFile(fn)
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
    rois = pd.read_csv(path, index_col=0)
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
    roi_props = {'roi_id': rois_at_pos['roi_id'].values}
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
    new_rois = pd.DataFrame(point_layer.data*downscale, columns=['frame','zc','yc', 'xc'])
    new_rois.index.name = 'roi_id'
    new_rois = new_rois.reset_index()
    new_rois['position'] = position
    new_rois['ch'] = rois['ch'].loc[0]

    rois = rois.drop(rois[rois['position']==position].index)
    return pd.concat([rois, new_rois]).sort_values('position')

def filter_rois_in_nucs(rois, nuc_masks, pos_list, new_col='nuc_label'):
    '''Check if a spot is in inside a segmented nucleus.

    Args:
        rois (DataFrame): ROI table to check
        nuc_masks (list): List of 2D nuclear mask images, where 0 is outside nuclei and >0 inside
        pos_list (list): List of all the positions (str) to check
        new_col (str, optional): The name of the new column in the ROI table. Defaults to 'nuc_label'.

    Returns:
        rois (DataFrame): Updated ROI table indicating if ROI is inside nucleus or not.
    '''


    if not nuc_masks:
        print('No nuclear masks provided, cannot filter.')
        return rois

    def spot_in_nuc(row, nuc_masks):
        pos_index = pos_list.index(row['position'])
        try:
            spot_label = nuc_masks[pos_index][int(row['yc']), int(row['xc'])]
        except IndexError: #If due to drift spot is outside frame.
            spot_label = 0
        return spot_label
    
    try:
        rois.drop(columns=[new_col], inplace=True)
    except KeyError:
        pass

    rois[new_col] = rois.apply(spot_in_nuc, nuc_masks=nuc_masks, axis=1)
    print('ROIs filtered.')
    return rois


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
    with czifile.CziFile(image_path) as czi:
        image=czi.asarray()[0,0,:,0,:,:,:,0]
    return image


def read_tif_image(image_path):
    with tifffile.TiffFile(image_path) as tif:
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
    
    with czifile.CziFile(image_path) as czi:
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

    
def detect_spots(img, spot_threshold=20):
    '''Spot detection by difference of gaussian filter
    #TODO: Do not use hard-coded sigma values

    Args:
        img (ndarray): Input 3D image
        spot_threshold (int): Threshold to use for spots. Defaults to 20.

    Returns:
        spot_props (DataFrame): The centroids and roi_IDs of the spots found. 
        img (ndarray): The DoG filtered image used for spot detection.
    '''

    img = gaussian(img, 0.8)-gaussian(img,1.3)
    img = (img-np.mean(img))/np.std(img)
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
    return spot_props, img
    #Cleanup and saving of the DataFrame
        
def drift_corr_course(t_img, o_img, downsample=1):
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
    course_drift=phase_cross_correlation(np.array(t_img[s]), np.array(o_img[s]), return_error=False) * downsample
    #Shift image for fine drift correction
    #o_img=ndi.shift(o_img,course_drift,order=0)
    print('Course drift:', course_drift)
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
    try:
        rand_points = t_img_maxima[np.random.choice(t_img_maxima.shape[0], size=n_points), :]
    except ValueError: #If no maxima are found just choose one random point:
        rand_points = [[10,10,10]]
    
    #Initialize array to store shifts for all selected fiducials.
    shifts=np.empty_like(rand_points, dtype=np.float32)
    
    #Calculate fine scale drift for all selected fiducials.
    sub_imgs_t = []
    sub_imgs_o = []
    for i, point in enumerate(rand_points):
        s_t = tuple([slice(ind-8, ind+8) for ind in point])
        s_o = tuple([slice(ind-int(shift)-8, ind-int(shift)+8) for (ind, shift) in zip(point, course_drift)])
        t = t_img[s_t]
        o = o_img[s_o]
        if (t.shape == (16, 16, 16)) and (o.shape == (16,16,16)):
            sub_imgs_t.append(t)
            sub_imgs_o.append(o)
        else:
            img = np.zeros((16, 16, 16))
            img[8,8,8] = 1000
            sub_imgs_t.append(img)
            sub_imgs_o.append(img)

    shifts = dask.compute([dask.delayed(phase_cross_correlation)(t, 
                                        o, 
                                        upsample_factor=upsampling,
                                        return_error=False)
                            for (t,o) in zip(sub_imgs_t, sub_imgs_o)])[0]
    fine_drift = trim_mean(shifts, proportiontocut=0.2, axis=0)
    print('Fine drift:', fine_drift)
    #Return the 60% central mean to avoid outliers.
    return fine_drift#, np.std(shifts, axis=0)

def napari_view(img, points=None, downscale=2, contrast_limits=(100,10000), point_frame_size = 1):
    with napari.gui_qt():
        if not isinstance(img, list):
            viewer = napari.view_image(img[...,::downscale,::downscale,::downscale], contrast_limits=contrast_limits)
        else:
            viewer = napari.view_image(img[0][...,::downscale,::downscale,::downscale], contrast_limits=contrast_limits)
            colors = ['green', 'magenta', 'grey']
            for i in img[1:]:
                viewer.add_image(i[...,::downscale,::downscale,::downscale], contrast_limits=contrast_limits)
        if points is not None:
            point_layer = viewer.add_points(points/downscale, 
                                                    size=(point_frame_size,15,15,15),
                                                    edge_width=3,
                                                    edge_color='red',
                                                    face_color='transparent',
                                                    n_dimensional=True)
            sel_dim = list(points[0,:]/downscale)
            for dim in range(len(sel_dim)):
                viewer.dims.set_current_step(dim, sel_dim[dim])

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

def nuc_segmentation(nuc_imgs, diameter = 150, do_3D = False):
    '''
    Runs nuclear segmentation using cellpose trained model (https://github.com/MouseLand/cellpose)

    Args:
        nuc_imgs (ndarray or list of ndarrays): 2D or 3D images of nuclei, expects single channel
    '''
    from cellpose import models
    model = models.Cellpose(gpu=False, model_type='nuclei')
    channels = [0,0]
    masks, flows, styles, diams = model.eval(nuc_imgs, diameter=diameter, channels=channels, net_avg=False, do_3D=do_3D)
    return masks

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