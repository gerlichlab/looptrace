"""Tools and values for dealing with how this package treats filepaths"""

import os
from pathlib import Path
from typing import *

from gertils.pathtools import ExtantFile, ExtantFolder

__author__ = "Vince Reuter"


SPOT_IMAGES_SUBFOLDER = "spot_images_dir"


def get_spot_images_path(folder: Union[str, Path, ExtantFolder]):
    """Provide the path to the fixed-name subfolder for the spot images, relative to the given folder."""
    return os.path.join(folder, SPOT_IMAGES_SUBFOLDER)


def simplify_path(p: Union[str, Path, ExtantFile, ExtantFolder, None]) -> Union[str, Path, None]:
    return p.path if isinstance(p, (ExtantFile, ExtantFolder)) else p
