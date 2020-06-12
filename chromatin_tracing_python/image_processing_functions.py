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
import numpy as np
from xml.etree import cElementTree as ElementTree
import h5py

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
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if all([s in file for s in template]):
                files.append(os.path.join(r, file))
    return sorted(files)

def match_file_lists(t_list,o_list):
    t_list_match = [item.split('__')[0][-5:] for item in t_list]
    o_list_match = [item for item in o_list if item.split('__')[0][-5:] in t_list_match]
    return o_list_match

def match_file_lists_decon(t_list,o_list):
    t_list_match = [item.split('\\')[-1].split('__')[0] for item in t_list]
    
    o_list_match = [item for item in o_list if 
                    item.split('\\')[-1].split('_P0001_')[0] in t_list_match]
    
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

    
    
