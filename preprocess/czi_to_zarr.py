import os
import re
import itertools
from numcodecs import Blosc
import zarr
import czifile
import joblib
import sys

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

def read_czi_image(image_path):
    '''
    Reads czi files as arrays using czifile package. Returns only CZYX image.
    '''
    with czifile.CziFile(image_path) as czi:
        image=czi.asarray()[0,0,:,0,:,:,:,0]
    return image
         
def single_czi_to_zarr(index, store, path):
    img = read_czi_image(path)
    store[index] = img

def czi_to_zarr(grouped_paths, out_path):
    compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
    sanple_shape = czifile.CziFile(grouped_paths[0][0]).shape
    X = sanple_shape[-2]
    Y = sanple_shape[-3]
    Z = sanple_shape[-4]
    C = sanple_shape[-6]
    T = len(grouped_paths[0])
    P = len(grouped_paths)
    s = (P, T, C, Z, Y, X)
    print('Found files of shape:', s)
    chunks = (1,1,1,1,s[-2],s[-1])
    z = zarr.open(out_path+os.sep+'images.zip', mode='w', compressor=compressor, shape=s, chunks=chunks)

    for i, pos in enumerate(grouped_paths):
        print(f'Saving position {i} of {len(grouped_paths)}.')
        joblib.Parallel(n_jobs=-1, prefer='threads', verbose=10)(joblib.delayed(single_czi_to_zarr)((i,j), z, path) for j, path in enumerate(pos))


if __name__ == '__main__':
    print('Starting.')
    try:
        template = list(sys.argv[2].split(','))
    except IndexError:
        template = ['DE_2','.czi']
    all_paths = all_matching_files_in_subfolders(path = sys.argv[1], template = template)
    print('Found number of paths:', len(all_paths))
    grouped_paths, positions = group_filelist(all_paths, re_phrase='W[0-9]{4}')
    print('Grouped paths into n groups:', len(positions))
    czi_to_zarr(grouped_paths, sys.argv[1])