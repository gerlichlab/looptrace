"""Tools and values for dealing with how this package treats filepaths"""

import os
from pathlib import Path
from typing import *

from gertils import ExtantFile, ExtantFolder

__author__ = "Vince Reuter"
__credits__ = ["Vince Reuter"]

__all__ = ["SPOT_IMAGES_SUBFOLDER", "FilePathLike", "FolderPathLike", "PathLike", "get_analysis_path", "simplify_path"]


FilePathLike = Union[str, Path, ExtantFile]
FolderPathLike = Union[str, Path, ExtantFolder]
PathLike = Union[FilePathLike, FolderPathLike]
SPOT_IMAGES_SUBFOLDER = "spot_images_dir"


def get_analysis_path(config: Mapping[str, Any]) -> str:
    return os.path.expanduser(os.path.expandvars(config["analysis_path"]))


def simplify_path(p: Optional[PathLike]) -> Optional[Path]:
    """
    Take a value that should be a Path, possbily null, and return the same value (fundamentally) but of narrower type.

    Parameters
    ----------
    p : str or Path or gertils.ExtantFile or gertils.ExtantFolder, optional
        The value to homogenise as an optional pathlib.Path

    Returns
    -------
    Optional[pathlib.Path]
        The pathlib.Path representation of the given value, or null if given object was null
    
    Raises
    ------
    TypeError : if the given value is not a member of one of the pathlike types, and is non-null
    """
    if p is None:
        return
    if isinstance(p, (ExtantFile, ExtantFolder)):
        return p.path
    if isinstance(p, Path):
        return p
    if isinstance(p, str):
        return Path(p)
    raise TypeError(f"Value of illegal input type ({type(p).__name__}) for path simplification: {p}")
