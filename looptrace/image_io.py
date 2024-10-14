"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import copy
from enum import Enum
import itertools
import logging
from operator import itemgetter
import os
from pathlib import Path
import re
import shutil
import time
from typing import *
import zipfile

import dask.array as da
import numcodecs
import numpy as np
import numpy.typing as npt
import tqdm
import zarr


POSITION_EXTRACTION_REGEX = r"(Point\d+)"
TIME_EXTRACTION_REGEX = r"(Time\d+)"


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
    def __init__(self, filepath: Union[str, Path]):
        self.npz = np.load(filepath, allow_pickle=True)
        self.files = self.npz.files
        self.filepath = filepath

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


def multi_ome_zarr_to_dask(folder: str, remove_unused_dims: bool = True):
    """
    The multi_ome_zarr_to_dask function takes a folder path and returns a list of dask arrays and a list of image folders.
    
    This is done by reading multiple dask images from a single folder.
    If the remove_unused_dims flag is set to True, the function will also remove unnecessary dimensions from the dask array.

    Parameters
    ----------
    folder : str
        Input folder path
    remove_unused_dims : bool, default True
        Whether to collapse each trivial (length-1) dimension, e.g. (1, 1, 1, 2044, 2048) -> (2044, 2048)
    
    Returns
    -------
    (list of da.core.Array, list of str)
        list of dask arrays of the images and list of strings of image folder names
    """
    print("Parsing zarr to dask: ", folder)
    image_folders = sorted([p.name for p in os.scandir(folder) if os.path.isdir(p) and not ignore_path(p)])
    out = []
    for image in image_folders:
        print("Parsing subfolder: ", image)
        curr_path = os.path.join(folder, image)
        path_to_open = curr_path if (Path(curr_path) / ".zarray").is_file() else os.path.join(curr_path, "0")
        z = zarr.open(path_to_open)
        try:
            arr = da.from_zarr(z)
        except zarr.core.ArrayNotFoundError:
            print(f"ERROR reading zarr array from {path_to_open}")
            raise
        # TODO: consider if this is wise!
        if remove_unused_dims:
            new_slice = tuple([0 if i == 1 else slice(None) for i in arr.shape])
            arr = arr[new_slice]
        out.append(arr)
    print(f"Loaded list of {len(out)} arrays.")
    return out, image_folders


def multipos_nd2_to_dask(folder: str):
    """
    The function takes a folder path and returns a list of dask arrays and a 
    list of image folders by reading multiple nd2 images each with multiple FOVs in a single folder.

    Args:
        folder (str): Input folder path

    Returns:
        list: list of dask arrays of the images
    """

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


def parse_fields_of_view_from_text(s: str) -> List[str]:
    return re.findall(POSITION_EXTRACTION_REGEX, s)


def parse_times_from_text(s: str) -> List[str]:
    return re.findall(TIME_EXTRACTION_REGEX, s)


def single_fov_to_zarr(
    *, 
    images: np.ndarray | list, 
    path: Union[str, Path],
    name: str, 
    fov_name: str, 
    dtype: Type, 
    axes = ('t','c',"z","y","x"), 
    chunk_axes = ("y", "x"), 
    chunk_split = (2,2),  
    metadata: dict = None,
    compressor: Optional[numcodecs.abc.Codec] = None,
    ):
    """
    Function to write a single FOV image with optional amount of additional dimensions to zarr.
    """
    def single_image_to_zarr(z: zarr.DirectoryStore, idx: str, img: np.ndarray):
        '''Small helper function.

        Args:
            z (zarr.DirectoryStore): Zarr store
            idx (str): (Time) index to write
            img (np.ndarray): image data to write
        '''
        z[idx] = img
    
    store = zarr.DirectoryStore(os.path.join(path, fov_name if fov_name.endswith(".zarr") else fov_name + ".zarr"))
    root = zarr.group(store=store, overwrite=True)

    size = {}
    chunk_dict = {}
    # TODO: handle better the absence of dimensions w.r.t. shape and chunks.
    # This is relevance, e.g., for NucDetector.generate_images_for_segmentation.
    # Namely, different readers may not like the fact that the shape and chunks don't match underlying data.
    # This can happen when one or more dimensions collapses down flat, to a trivial single dimension.
    # See: https://github.com/gerlichlab/looptrace/issues/245
    default_axes = ("t", "c", "z", "y", "x")
    try:
        print(f"Building metdata for image with shape {images.shape}")
    except AttributeError:
        pass
    for ax in default_axes:
        sz: int
        ck: int
        if ax in axes:
            # TODO: fix the signature or implementation of this, as well as call sites, since 
            #       if images argument is a list, it won't have a .shape attribute to access.\
            sz = images.shape[axes.index(ax)]
            ck = sz // chunk_split[chunk_axes.index(ax)] if ax in chunk_axes else 1
        else:
            sz = 1
            ck = 1
        size[ax] = sz
        chunk_dict[ax] = ck
    
    print(f"Shape metadata: {size}")
    print(f"Shape metadata: {chunk_dict}")

    shape = tuple([size[ax] for ax in default_axes])
    chunks = tuple([chunk_dict[ax] for ax in default_axes])
    images = np.reshape(images, shape)

    root.attrs["multiscale"] = {
        "multiscales": [{
            "version": "0.3", 
            "name": name + "_" + fov_name, 
            "datasets": [{"path": "0"}],
            "axes": ["t","c","z","y","x"],
        }]
    }
    if metadata:
        root.attrs["metadata"] = metadata

    compressor = compressor or numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)

    multiscale_level = root.create_dataset(name=str(0), compressor=compressor, shape=shape, chunks=chunks, dtype=dtype)
    if "t" in chunk_axes:
        multiscale_level[:] = images
    elif size["t"] < 10 or images.size < 1e9:
        for i in range(size["t"]):
            single_image_to_zarr(multiscale_level, i, images[i])
    else:
        import joblib
        joblib.Parallel(n_jobs=-1, prefer="threads", verbose=10)(joblib.delayed(single_image_to_zarr)
                                                            (multiscale_level, i, images[i]) for i in range(size["t"]))


def nuc_multipos_single_time_max_z_proj_zarr(
    name_img_pairs: List[Tuple[str, np.ndarray]], 
    root_path: str, 
    dtype: Type, 
    metadata: Optional[dict] = None,
    overwrite: bool = True,
    ):
    if not isinstance(name_img_pairs, (list, tuple)):
        raise TypeError(f"Sequence of pairs of name and image data must be list or tuple, not {type(name_img_pairs).__name__}")
    axes = ("y", "x")
    bad_name_shape_pairs = [(name, img.shape) for name, img in name_img_pairs if len(img.shape) != len(axes)]
    if bad_name_shape_pairs:
        raise ValueError(f"{len(bad_name_shape_pairs)}/{len(name_img_pairs)} images with bad shape given {len(axes)} axes: {bad_name_shape_pairs}")
    write_jvm_compatible_zarr_store(
        name_data_pairs=[(name if name.endswith(".zarr") else name + ".zarr", img) for name, img in name_img_pairs], 
        root_path=root_path, 
        dtype=dtype, 
        metadata=metadata, 
        overwrite=overwrite,
        )


def write_jvm_compatible_zarr_store(
    name_data_pairs: Sequence[Tuple[str, np.ndarray]], 
    root_path: Union[str, Path], 
    dtype: Type, 
    metadata: Optional[dict] = None,
    overwrite: bool = False,
    ) -> List[Path]:
    if not name_data_pairs:
        raise ValueError("To write data to zarr, a nonempty sequence of name/data pairs must be passed!")
    print(f"INFO: creating zarr store rooted at {root_path}")
    store = zarr.DirectoryStore(root_path)
    root = zarr.group(store=store, overwrite=overwrite)
    if metadata:
        root.attrs["metadata"] = metadata
    paths = []
    for name, data in name_data_pairs:
        # Use numcodecs.ZLib() as the compressor to ensure readability by cdm-core / netcdf-Java.
        dataset = root.create_dataset(name=name, compressor=numcodecs.Zlib(), shape=data.shape, dtype=dtype)
        dataset[:] = data
        paths.append(Path(root_path) / name)
    return paths


def images_to_ome_zarr(
    *, 
    name_image_pairs: Iterable[npt.ArrayLike], 
    path: Union[str, Path], 
    data_name: str, 
    axes = ("t", "c", "z", "y", "x"), 
    chunk_axes = ("y", "x"), 
    chunk_split = (2, 2),  
    metadata: dict = None,
):
    """
    Saves an array to ome-zarr format (metadata still needs work to match spec)
    """
    if not isinstance(axes, tuple) or not all(isinstance(a, str) for a in axes):
        raise TypeError("axes argument must be tuple[str])")
    if "p" in axes:
        raise ValueError(
            "Position/FOV axis 'p' was supplied, but images_to_ome_zarr assumes each image is for a field of view."
        )
    
    if not os.path.isdir(path):
        os.makedirs(path)

    bit_depth: PixelArrayBitDepth = PixelArrayBitDepth.get_unique_bit_depth((img for _, img in name_image_pairs))
    logging.info(f"Will save OME ZARR with bit depth: {bit_depth}")

    for fov_name, fov_img in name_image_pairs:
        single_fov_to_zarr(
            images=fov_img, 
            path=path, 
            name=data_name, 
            fov_name=fov_name, 
            dtype=bit_depth.value, 
            axes=axes, 
            chunk_axes=chunk_axes, 
            chunk_split=chunk_split, 
            metadata=metadata,
            )

    print("OME ZARR images generated.")


def create_zarr_store(  path: Union[str, Path],
                        name: str, 
                        fov_name: str,
                        shape: tuple, 
                        dtype: str,  
                        chunks: tuple,   
                        metadata: Optional[dict] = None,
                        voxel_size: list = [1, 1, 1]):
    store = zarr.NestedDirectoryStore(os.path.join(path, fov_name))
    root = zarr.group(store=store, overwrite=True)

    root.attrs["multiscales"] = [{"version": "0.4", 
                                    "name": name + "_" + fov_name, 
                                    "datasets": [{"path": "0",                     
                                                    "coordinateTransformations": [{"type": "scale",
                                                                                    "scale": [1.0, 1.0] + voxel_size}]},],
                                    "axes": [
                                        {"name": "t", "type": "time", "unit": "minute"},
                                        {"name": "c", "type": "channel"},
                                        {"name": "z", "type": "space", "unit": "micrometer"},
                                        {"name": "y", "type": "space", "unit": "micrometer"},
                                        {"name": "x", "type": "space", "unit": "micrometer"}],}]
    if metadata is not None:
        root.attrs["metadata"] = metadata

    compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)

    level_store = root.create_dataset(name=str(0), compressor=compressor, shape=shape, chunks=chunks, dtype=dtype)
    return level_store


def zip_folder(folder, out_file, compression = zipfile.ZIP_STORED, remove_folder=False, retry_if_fails: bool = True) -> str:
    #Zips the contents of a folder and stores as filename in same dir as folder.
    #Strips the original fileextensions (usecase for npy -> npz archive). Will probably modify this in future.
    if remove_folder and os.path.dirname(out_file) == folder:
        raise ValueError(f"Cannot zip to file ({out_file}) in folder to be deleted ({folder})")
    filelist = sorted([(p.path, p.name) for p in os.scandir(folder)], key=itemgetter(0))
    with zipfile.ZipFile(out_file, mode="w", compression=compression, compresslevel=3) as zfile:
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
    """
    Parameters
    ----------
    path : String with file path to h5 file.
    index : Tuple with slice indexes of h5 file. Assumed CTZYX order.
            Default is all slices.
    
    Returns
    -------
    Image as numpy array.
    """
    import h5py    
    with h5py.File(path, "r") as f:
        if ch is not None:
            index=(slice(ch,ch+1),slice(None),)+index
            img=f[list(f.keys())[0]]["ImageData"]["Image"][index][()][0,0]
        else:
            index=(slice(None),slice(None))+index
            img=f[list(f.keys())[0]]["ImageData"]["Image"][index][()][:,0]
        
    return img


def all_matching_files_in_subfolders(path, template):
    """
    Generates a sorted list of all files with the template in the 
    filename in directory and subdirectories.
    """

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if all([s in file for s in template]):
                files.append(os.path.join(r, file))
    return sorted(files)

def group_filelist(input_list, re_phrase):
    """
    Takes a list of strings (typically filepaths) and groups them according
    to a given element given by its FOV after splitting the string at split_char.
    E.g.for '..._WXXXX_PXXXX_TXXXX.ext' format this will by split_char='_' and element = -3.
    Returns a list of the , and 
    """
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


class PixelArrayBitDepth(Enum):
    MonoByte = np.uint8
    DiByte = np.uint16

    @classmethod
    def for_array(cls, arr: npt.ArrayLike) -> Optional["PixelArrayBitDepth"]:
        try:
            return cls.for_array(arr)
        except (AttributeError, TypeError, ValueError):
            return None
    
    @classmethod
    def unsafe_for_array(cls, arr: npt.ArrayLike) -> "PixelArrayBitDepth":
        # First, check that the input is well typed.
        dtype = arr.dtype
        if dtype not in [int, np.uint8, np.uint16, np.uint32, np.uint64]:
            raise TypeError(f"Datatype for alleged pixel array isn't unsigned-int-like: {dtype}")
        
        max_pix = arr.max()
        min_pix = arr.min()
        if min_pix < 0:
            raise ValueError(f"Negative min value in alleged pixel array: {min_pix}")
        if max_pix < 256:
            return cls.MonoByte
        if max_pix < 65536:
            return cls.DiByte

        raise ValueError(f"Could not determine bit depth for pixel array with max value {max_pix}")
    
    @classmethod
    def get_unique_bit_depth(cls, arrays: Iterable[npt.ArrayLike]) -> "PixelArrayBitDepth":
        depths: set["PixelArrayBitDepth"] = {cls.unsafe_for_array(arr) for arr in arrays}
        if len(depths) > 1:
            raise RuntimeError(f"Multiple ({len(depths)}) bit depths determined for masks: {', '.join(d.name for d in depths)}")
        try:
            return next(iter(depths))
        except StopIteration as e:
            raise ValueError("No bit depth determined; was collection of arrays empty?") from e


class ImageParseException(Exception):
    """Error subtype for when at least one error occurs during image parse"""
    
    def __init__(self, errors: Dict[str, Exception]) -> None:
        if len(errors) == 0:
            raise ValueError("Errors must be nonempty to create an exception.")
        msg = f"{len(errors)} error(s) during image parsing: {errors}"
        super().__init__(msg)
        self._errors = errors
    
    @property
    def errors(self):
        return copy.deepcopy(self._errors)
