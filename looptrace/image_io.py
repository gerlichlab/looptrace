"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import copy
import itertools
from operator import itemgetter
import os
from pathlib import Path
import re
import shutil
import time
from typing import *
import zipfile

import dask
import dask.array as da
import numcodecs
import numpy as np
import tqdm
import zarr

from looptrace.integer_naming import get_position_name_1


POSITION_EXTRACTION_REGEX = r"(Point\d+)"
TIME_EXTRACTION_REGEX = r"(Time\d+)"
TIFF_EXTENSIONS = [".tif", ".tiff"]


def ignore_path(p: Union[str, os.DirEntry, Path]) -> bool:
    try:
        name = p.name
    except AttributeError:
        name = p
    return name.startswith("_")


class NPZ_wrapper():
    '''
    Class wrapping the numpy .npz loader to allow slicing npz files as standard arrays
    Note that this returns a list of arrays in case the stacks have different dimensions.
    '''
    def __init__(self, fid):
        #super().__init__(fid, allow_pickle=True)
        self.npz = np.load(fid, allow_pickle=True)
        self.files = self.npz.files

    def __iter__(self):
        return iter(self.npz[f] for f in self.files)

    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, i):
        if isinstance(i, str):
            return self.npz[i]
        elif isinstance(i, int):
            return self.npz[self.files[i]]
        elif isinstance(i, tuple):
            return [a[i[1:]] for a in self.return_npz_slice(i[0])]
        elif isinstance(i, slice):
            return self.return_npz_slice(i)
        elif isinstance(i, list) or isinstance(i, np.ndarray):
            return [self.npz[self.files[j]] for j in i] 
    
    def return_npz_slice(self, s):
        if not isinstance(s, slice):
            s = slice(s)
        return [self.npz[self.files[j]] for j in list(range(len(self.files))[s])]


# class RaggedArray():
#     '''
#     Class for allowing slicing of a list of ragged arrays as if it was one large array.
#     NB! Does not work yet, slicing is incorrect!
#     '''
#     def __init__(self, arr):
#         self.arr = arr

#     def __getitem__(self, i):
#         if isinstance(i, int):
#             return self.arr[i]
#         elif isinstance(i, tuple):
#             return [a[i[1:]] for a in self.return_list_slice(i[0])]
#         elif isinstance(i, slice):
#             return self.return_list_slice(i)
    
#     def return_list_slice(self, s):
#         if not isinstance(s, slice):
#             s = slice(s)
#         return [self.arr[j] for j in list(range(len(self.arr)+1)[s])]


def multi_ome_zarr_to_dask(folder: str, remove_unused_dims = True):
    '''The multi_ome_zarr_to_dask function takes a folder path and returns a list of dask arrays and a list of image folders by reading multiple dask images in a single folder.
        If the remove_unused_dims flag is set to True, the function will also remove unnecessary dimensions from the dask array.

    Args:
        folder (str): Input folder path
        remove_unused_dims (bool, optional): Defaults to True.

    Returns:
        list: list of dask arrays of the images
        list: list of strings of image folder names
    '''
    print("Parsing zarr to dask: ", folder)
    image_folders = sorted([p.name for p in os.scandir(folder) if os.path.isdir(p) and not ignore_path(p)])
    out = []
    for image in image_folders:
        print("Parsing subfolder: ", image)
        z = zarr.open(os.path.join(folder, image, '0'))
        arr = da.from_zarr(z)

        # Remove unecessary dimensions: #TODO consider if this is wise!
        if remove_unused_dims:
            new_slice = tuple([0 if i == 1 else slice(None) for i in arr.shape])
            arr = arr[new_slice]
        #chunks = (1,1,s[-3], s[-2], s[-1])
        out.append(arr)#, chunks=chunks))
    #out = da.stack(out) #Removed as not compatible with different shaped
    print('Loaded list of ', len(out), ' arrays.')
    return out, image_folders

def multipos_nd2_to_dask(folder: str):
    '''The function takes a folder path and returns a list of dask arrays and a 
    list of image folders by reading multiple nd2 images each with multiple positions in a single folder.

    Args:
        folder (str): Input folder path

    Returns:
        list: list of dask arrays of the images
    '''

    import nd2
    image_files = sorted([p.path for p in os.scandir(folder) if p.name.endswith('.nd2')])
    out = []
    #print(image_folders)
    for image in tqdm.tqdm(image_files):
        with nd2.ND2File(image, validate_frames = False) as imgdat:
            out.append(imgdat.to_dask())
    out = da.stack(out)
    out = da.moveaxis(out, 2, 3)
    out = da.moveaxis(out, 0, 1)
    #print('Loaded nd2 arrays of shape ', out.shape)
    return out


def parse_positions_from_text(s: str) -> List[str]:
    return re.findall(POSITION_EXTRACTION_REGEX, s)


def parse_times_from_text(s: str) -> List[str]:
    return re.findall(TIME_EXTRACTION_REGEX, s)


def stack_tif_to_dask(folder: str):
    '''The function takes a folder path and returns a list of dask arrays and a 
    list of image folders by reading multiple tif images where each represents a 3D stack (split by position and time) in a single folder.

    Args:
        folder (str): Input folder path

    Returns:
        list: list of dask arrays of the images
    '''
    import tifffile
    
    image_files = sorted([p.path for p in os.scandir(folder) if _has_tiff_extension(p)])
    try:
        image_times = sorted(list(set([parse_times_from_text(s)[0] for s in image_files])))
    except IndexError:
        time_dim = False
    else:
        time_dim = True
    image_points = sorted(list(set([parse_positions_from_text(s)[0] for s in image_files])))
    out = []
    for p in tqdm.tqdm(image_points):
        if time_dim:
            t_stack = []
            for t in image_times:
                path = list(filter(lambda s: (p in s) and (t in s), image_files))[0]
                arr = tifffile.memmap(path)
                arr = da.moveaxis(arr, 0, 1)
                t_stack.append(arr)
        else:
            path = list(filter(lambda s: (p in s), image_files))[0]
            arr = tifffile.memmap(path)
            arr = da.moveaxis(arr, 0, 1)
            t_stack = [arr]
        out.append(da.stack(t_stack))
    return out, image_points


def single_position_to_zarr(
    images: np.ndarray or list, 
    path: str,
    name: str, 
    pos_name: str, 
    dtype: Type, 
    axes = ('t','c','z','y','x'), 
    chunk_axes = ('y', 'x'), 
    chunk_split = (2,2),  
    metadata: dict = None,
    compressor: Optional[numcodecs.abc.Codec] = None,
    ):
    '''
    Function to write a single position image with optional amount of additional dimensions to zarr.
    '''
    def single_image_to_zarr(z: zarr.DirectoryStore, idx: str, img: np.ndarray):
        '''Small helper function.

        Args:
            z (zarr.DirectoryStore): Zarr store
            idx (str): (Time) index to write
            img (np.ndarray): image data to write
        '''
        z[idx] = img
    
    store = zarr.DirectoryStore(os.path.join(path, pos_name + ".zarr"))
    root = zarr.group(store=store, overwrite=True)

    size = {}
    chunk_dict = {}
    # TODO: handle better the absence of dimensions w.r.t. shape and chunks.
    # This is relevance, e.g., for NucDetector.gen_nuc_images.
    # Namely, different readers may not like the fact that the shape and chunks don't match underlying data.
    # This can happen when one or more dimensions collapses down flat, to a trivial single dimension.
    # See: https://github.com/gerlichlab/looptrace/issues/245
    default_axes = ('t','c','z','y','x')
    try:
        print(f"Building metdata for image with shape {images.shape}")
    except AttributeError:
        pass
    for ax in default_axes:
        if ax in axes:
            # TODO: fix the signature or implementation of this, as well as call sites, since 
            #       if images argument is a list, it won't have a .shape attribute to access.
            size[ax] = images.shape[axes.index(ax)]
            if ax in chunk_axes:
                chunk_dict[ax] = size[ax]//chunk_split[chunk_axes.index(ax)]
            else:
                chunk_dict[ax] = 1
        else:
            size[ax] = 1
            chunk_dict[ax] = 1
    
    print(f"Shape metadata: {size}")
    print(f"Shape metadata: {chunk_dict}")

    shape = tuple([size[ax] for ax in default_axes])
    chunks = tuple([chunk_dict[ax] for ax in default_axes])
    images = np.reshape(images, shape)

    root.attrs['multiscale'] = {
        'multiscales': [{
            'version': '0.3', 
            'name': name + '_' + pos_name, 
            'datasets': [{'path': '0'}],
            'axes': ['t','c','z','y','x'],
        }]
    }
    if metadata:
        root.attrs['metadata'] = metadata

    compressor = compressor or numcodecs.Blosc(cname='zstd', clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)

    multiscale_level = root.create_dataset(name=str(0), compressor=compressor, shape=shape, chunks=chunks, dtype=dtype)
    if 't' in chunk_axes:
        multiscale_level[:] = images
    elif size['t'] < 10 or images.size < 1e9:
        for i in range(size['t']):
            single_image_to_zarr(multiscale_level, i, images[i])
    else:
        import joblib
        joblib.Parallel(n_jobs=-1, prefer='threads', verbose=10)(joblib.delayed(single_image_to_zarr)
                                                            (multiscale_level, i, images[i]) for i in range(size['t']))


def nuc_multipos_single_time_max_z_proj_zarr(name_img_pairs: List[Tuple[str, np.ndarray]], root_path: str, dtype: Type, metadata: dict = None):
    axes = ('y', 'x')
    bad_name_shape_pairs = [(name, img.shape) for name, img in name_img_pairs if len(img.shape) != len(axes)]
    if bad_name_shape_pairs:
        raise ValueError(f"{len(bad_name_shape_pairs)}/{len(name_img_pairs)} images with bad shape given {len(axes)} axes: {bad_name_shape_pairs}")
    store = zarr.DirectoryStore(root_path)
    root = zarr.group(store=store, overwrite=True)
    if metadata:
        root.attrs["metadata"] = metadata
    for pos_name, img in name_img_pairs:
        dataset = root.create_dataset(name=pos_name + ".zarr", compressor=numcodecs.Zlib(), shape=img.shape, dtype=dtype)
        dataset[:] = img


def images_to_ome_zarr(
    images: np.ndarray or list, 
    path: str, 
    name: str, 
    dtype: Type, 
    axes = ('p','t','c','z','y','x'), 
    chunk_axes = ('y', 'x'), 
    chunk_split = (2,2),  
    metadata: dict = None,
    ):
    '''
    Saves an array to ome-zarr format (metadata still needs work to match spec)
    '''
    # TODO: type/shape/structure of images should correspond to conditional here, about axes.

    if not os.path.isdir(path):
        os.makedirs(path)

    if 'p' in axes:
        for i, pos_img in enumerate(images):
            pos_name = get_position_name_1(i)
            single_position_to_zarr(pos_img, path, name, pos_name, dtype, axes[1:], chunk_axes, chunk_split, metadata)
    else:
        # TODO: fix the fact that here pos_name is undefined.
        single_position_to_zarr(images, path, name, pos_name, dtype, axes, chunk_axes, chunk_split, metadata)


    print('OME ZARR images generated.')

def create_zarr_store(  path: Union[str, Path],
                        name: str, 
                        pos_name: str,
                        shape: tuple, 
                        dtype: str,  
                        chunks: tuple,   
                        metadata: Optional[dict] = None,
                        voxel_size: list = [1, 1, 1]):
    store = zarr.NestedDirectoryStore(os.path.join(path, pos_name))
    root = zarr.group(store=store, overwrite=True)

    root.attrs['multiscales'] = [{'version': '0.4', 
                                    'name': name + '_' + pos_name, 
                                    'datasets': [{'path': '0',                     
                                                    "coordinateTransformations": [{"type": "scale",
                                                                                    "scale": [1.0, 1.0] + voxel_size}]},],
                                    "axes": [
                                        {"name": "t", "type": "time", "unit": "minute"},
                                        {"name": "c", "type": "channel"},
                                        {"name": "z", "type": "space", "unit": "micrometer"},
                                        {"name": "y", "type": "space", "unit": "micrometer"},
                                        {"name": "x", "type": "space", "unit": "micrometer"}],}]
    if metadata is not None:
        root.attrs['metadata'] = metadata

    compressor = numcodecs.Blosc(cname='zstd', clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)

    level_store = root.create_dataset(name=str(0), compressor=compressor, shape=shape, chunks=chunks, dtype=dtype)
    return level_store


def zip_folder(folder, out_file, compression = zipfile.ZIP_STORED, remove_folder=False, retry_if_fails: bool = True) -> str:
    #Zips the contents of a folder and stores as filename in same dir as folder.
    #Strips the original fileextensions (usecase for npy -> npz archive). Will probably modify this in future.
    if remove_folder and os.path.dirname(out_file) == folder:
        raise ValueError(f"Cannot zip to file ({out_file}) in folder to be deleted ({folder})")
    filelist = sorted([(p.path, p.name) for p in os.scandir(folder)], key=itemgetter(0))
    with zipfile.ZipFile(out_file, mode='w', compression=compression, compresslevel=3) as zfile:
        for f, fn in tqdm.tqdm(filelist, total=len(filelist)):
            zfile.write(f, arcname=os.path.splitext(fn)[0])
    if remove_folder:
        try:
            shutil.rmtree(folder)
        except (OSError, FileNotFoundError):
            if not retry_if_fails:
                raise
            time.sleep(1)
            try:
                shutil.rmtree(folder)
            except OSError as e:
                if 0 == len(os.listdir(folder)):
                    print(f"WARNING -- could not remove folder ({folder}) whose contents were zipped: {e}")
                else:
                    raise
    return out_file

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
    import h5py    
    with h5py.File(path, 'r') as f:
        if ch is not None:
            index=(slice(ch,ch+1),slice(None),)+index
            img=f[list(f.keys())[0]]['ImageData']['Image'][index][()][0,0]
        else:
            index=(slice(None),slice(None))+index
            img=f[list(f.keys())[0]]['ImageData']['Image'][index][()][:,0]
        
    return img


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

def read_czi_image(image_path):
    '''
    Reads czi files as arrays using czifile package. Returns only CZYX image.
    '''
    import czifile
    with czifile.CziFile(image_path) as czi:
        image=czi.asarray()[0,0,:,0,:,:,:,0]
    return image


def read_tif_image(image_path):
    import tifffile
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
    import czifile
    import yaml
    from xml import ElementTree

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

def czi_to_tif(in_folder, template, out_folder, prefix):
    '''Convert CZI files from MyPIC experiment to single YX tif images.

    Args:
        in_folder (str): Top level folder path to find czi files
        template (list): Template to match files to
        out_folder (str): Output folder to save tif images
        prefix (str): Prefix of output files, prepended to axis info
    '''
    import czifile
    import joblib
    import tifffile

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
    import tifffile
    imgs = []
    all_files = all_matching_files_in_subfolders(folder, ['.tif'])
    groups, positions = group_filelist(all_files, re_search)
    for i, group in enumerate(groups):
        print('Loading images for position ', positions[i])
        seq = tifffile.TiffSequence(group, pattern='axes')
        with seq.aszarr() as store:
            imgs.append(da.from_array(zarr.open(store, mode='r'), chunks =  (1,1,1,1,-1,-1))[0])
    return imgs
        
def nikon_tiff_to_dask(folder):
    import tifffile
    image_sequence = tifffile.TiffSequence(folder+os.sep+'*.tiff', pattern='(Time)(\d+)_(Point)(\d+)_(ZStack)(\d+)')
    print('Found image folder of shape', image_sequence.shape)
    with image_sequence.aszarr() as store:
        z = zarr.open(store, mode='r')
        print('Zarr shape: ', z.shape)
        images = da.transpose(da.from_zarr(z), (1,0,3,2,4,5))
    return images

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
    if '.czi' in template or _template_matches_tiff(template):
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
    elif _template_matches_tiff(template):
        sample = read_tif_image(all_files[0])
    else:
        raise TypeError('Input filetype not yet implemented.')

    pos_stack=[]
    for g in grouped_files:
        dask_arrays = []
        for fn in g:
            if '.czi' in template:
                d = dask.delayed(read_czi_image)(fn)
            elif _template_matches_tiff(template):
                d = dask.delayed(read_tif_image)(fn)
            array = da.from_delayed(d, shape=sample.shape, dtype=sample.dtype)
            dask_arrays.append(array)
        pos_stack.append(da.stack(dask_arrays, axis=0))
    #x = da.stack(pos_stack, axis=0)
    
    return pos_stack, groups, all_files


class ImageParseException(Exception):
    """Error subtype for when at least one error occurs during image """
    
    def __init__(self, errors: Dict[str, Exception]) -> None:
        if len(errors) == 0:
            raise ValueError("Errors must be nonempty to create an exception.")
        msg = f"{len(errors)} error(s) during image parsing: {errors}"
        super().__init__(msg)
        self._errors = errors
    
    @property
    def errors(self):
        return copy.deepcopy(self._errors)


def _has_tiff_extension(p: Union[str, Path]) -> bool:
    _, ext = os.path.splitext(p)
    return ext in TIFF_EXTENSIONS


def _template_matches_tiff(template: str) -> bool:
    return any(ext in template for ext in TIFF_EXTENSIONS)
