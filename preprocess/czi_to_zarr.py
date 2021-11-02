import os
import re
import itertools
from numcodecs import Blosc
import zarr
import czifile
import joblib
import sys
import json
import yaml
import shutil
import glob
import tqdm
import time
from nd2reader import ND2Reader

def all_matching_files_in_subfolders(path, template = ['DE_2','.czi']):
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

def group_filelist(input_list, re_phrase='W[0-9]{4}'):
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

def organize_czi_multipos_folder(folder):
    for root, dirs, files in os.walk(folder):
        for fname in files:
            # Match any string followed by a number in parethesis before .czi (Zen autosave multipos files)
            match_object = re.match('.+\((\d+)\).czi', fname)

            if match_object is None:
                # The regular expression did not match, ignore the file.
                continue

            # Form the new directory path using the number from the regular expression and the current root.
            new_dir = os.path.join(root, 'P'+match_object.group(1).zfill(4))

            if not os.path.isdir(new_dir):
                os.mkdir(new_dir)

            new_file_path = os.path.join(new_dir, fname)

            old_file_path = os.path.join(root, fname)
            shutil.move(old_file_path, new_file_path)

def read_czi_image(image_path: str, flip_dapi: bool = False):
    '''
    Reads czi files as arrays using czifile package. Returns only CZYX image.
    '''
    with czifile.CziFile(image_path) as czi:
        image=czi.asarray()[0,0,0,:,0,:,:,:,0]
    return image
         
def single_czi_to_zarr(index, z, path, flip_ch_idx):
    img = read_czi_image(path)
    #if index in flip_ch_idx:
        #img = img[1:, :, :, :]
    #    img = img[::-1, :, :, :]
    for i in range(3):
        try:
            z[index] = img
            break
        except OSError: #Network drive fail
            time.sleep(1)

def czi_to_omezarr_folder(input_folders: list, out_path: str, flip_ch_idx: tuple = (-1,), continue_from = None):
    '''[summary]

    Args:
        input_folders (list): [Input folders (shuold still be a list even if only one.)]
        out_path (str): [Destination folder for zarr images]
        flip_ch (tuple, optional): [Indicate timepoint(s) (index from 0) of data where channels should be reversed compared to other input data.]. Defaults to ().
    '''
    
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
        
    filedict = {}
    positions = [d for d in os.listdir(input_folders[0]) if os.path.isdir(os.path.join(input_folders[0], d))]

    for pos in positions:
        file_paths = []
        for folder_path in input_folders:
            file_paths.extend(sorted([f.path for f in os.scandir(folder_path+os.sep+pos)]))
        filedict[pos] = file_paths
    
    compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
    sample_path = filedict[positions[0]][-1]
    save_czi_sample_meta(sample_path, out_path)
    sample_shape = czifile.CziFile(sample_path).shape
    X = sample_shape[-2]
    Y = sample_shape[-3]
    Z = sample_shape[-4]
    C = sample_shape[-6]
    T = len(filedict[positions[0]])
    s = (T, C, Z, Y, X)

    print('Found files of shape:', s)
    chunks = (1,1,1,s[-2],s[-1])

    for k, pos in enumerate(tqdm.tqdm(positions)):
        if continue_from is not None: 
            continue_from_index = positions.index(continue_from)
            if (k < continue_from_index):
                print(continue_from_index, k)
                continue

        store = zarr.DirectoryStore(out_path+os.sep+pos+'.zarr')
        root = zarr.group(store=store, overwrite=True)

        with open(r"P:\MitoTrace\2021-08-11_MT001_2D_10pos_tracing\metadict.json") as f:
            root.attrs['omero'] = json.load(f)

        with open(r"C:\Git\looptrace_dev\preprocess\multiscales_template.json") as f:
            root.attrs['multiscale'] = json.load(f)

            multiscale_level = root.create_dataset(name = str(0), compressor=compressor, shape=s, chunks=chunks)
            joblib.Parallel(n_jobs=-1, prefer='threads', verbose=10)(joblib.delayed(single_czi_to_zarr)(i, multiscale_level, path, flip_ch_idx) for i, path in enumerate(filedict[pos]))

def save_czi_sample_meta(sample_path, out_path):
    shutil.copy2(sample_path, out_path+os.sep+'sample.czi')
    img = czifile.CziFile(sample_path)
    metadict = img.metadata(raw=False)
    with open(out_path+os.sep+'raw_image_metadata.yaml', 'w') as file:
        yaml.safe_dump(metadict, file)


def datetime_to_str(o):
    import datetime
    if isinstance(o, datetime.datetime):
        return o.__str__()

def multipos_nd2_to_zarr(folder_path, out_path):
    filelist = glob.glob(folder_path+os.sep+'*.nd2')
    
    compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
    with ND2Reader(filelist[0]) as images:
        sample_shape = images.sizes
        X = sample_shape['x']
        Y = sample_shape['y']
        Z = sample_shape['z']
        C = sample_shape['c']
        T = len(filelist)
        P = sample_shape['v']
        s = (T, C, Z, Y, X)

        meta = json.loads(json.dumps(images.metadata, default = datetime_to_str))

    pos_list = ['P'+str(p).zfill(4) for p in range(P)]
    print('Found files of shape:', s)
    chunks = (1,1,1,s[-2],s[-1])

    for i, pos in enumerate(tqdm.tqdm(pos_list)):
        store = zarr.DirectoryStore(out_path+os.sep+pos+'.zarr')
        root = zarr.group(store=store, overwrite=True)
        root.attrs['omero'] = meta
        with open(r"C:\Git\looptrace_dev\preprocess\multiscales_template.json") as f:
            root.attrs['multiscale'] = json.load(f)
        multiscale_level = root.create_dataset(name = str(0), compressor=compressor, shape=s, chunks=chunks)
        for j, f in enumerate(filelist):
            with ND2Reader(f) as images:
                images.bundle_axes = 'czyx'
                images.iter_axes = 'v'
                multiscale_level[j] = images[i]


def single_nd2_to_zarr(file_path, out_path, continue_from = None):
    
    compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
    with ND2Reader(file_path) as images:
        sample_shape = images.sizes
        X = sample_shape['x']
        Y = sample_shape['y']
        Z = sample_shape['z']
        C = sample_shape['c']
        T = sample_shape['t']
        P = sample_shape['v']
        s = (P, T, C, Z, Y, X)

        meta = json.loads(json.dumps(images.metadata, default = datetime_to_str))

        pos_list = ['P'+str(p+1).zfill(4) for p in range(P)]
        print('Found files of shape:', s)
        chunks = (1,1,1,Y,X)

        for i, pos in enumerate(tqdm.tqdm(pos_list)):

            if continue_from is not None: 
                continue_from_index = pos_list.index(continue_from)
                if (i < continue_from_index):
                    print(continue_from_index, i)
                    continue

            store = zarr.DirectoryStore(out_path+os.sep+pos+'.zarr')
            root = zarr.group(store=store, overwrite=True)
            root.attrs['omero'] = meta
            #with open(r"C:\Git\looptrace_dev\preprocess\multiscales_template.json") as f:
            #    root.attrs['multiscale'] = json.load(f)
            multiscale_level = root.create_dataset(name = str(0), compressor=compressor, shape=s[1:], chunks=chunks)
            for t in tqdm.tqdm(range(T)):
                images.bundle_axes = 'czyx'
                images.iter_axes = 'vt'
                multiscale_level[t] = images[(i*T)+t]