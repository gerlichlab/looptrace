"""Analysis of spot detection gridsearch"""

import os
from pathlib import Path
from typing import *
import pandas as pd

from looptrace.pathtools import ExtantFile, ExtantFolder
from detect_spots import Parameters
from detect_spots_gridsearch import ParamIndex


def read_config_file(fp: Union[str, Path]) -> Parameters:
    fp = ExtantFile(fp) if isinstance(fp, Path) else ExtantFile.from_string(fp)
    return Parameters.from_json_file(fp)


def read_config_directory(directory: ExtantFolder) -> Dict[str, Parameters]:
    result = {}
    for fn in os.listdir(directory.path):
        if os.path.splitext(fn)[1] != ".json":
            continue
        fp = directory.path / fn
        params = read_config_file(fp)
        result[fn] = params
    return result





my_gridsearch_folder = ExtantFolder.from_string("/groups/gerlich/experiments/Experiments_005800/005831/2023-06-20__spot_detection_runs/gridsearch2")

param_by_config = read_config_directory(my_gridsearch_folder)

FRAME_GROUPS = {
    "singleton": [2, 8, 11, 13], 
    "regional": [19, 20, 21, 22]
}




def get_frame_group(group: List[str]) -> str:
    return next(k for k, v in FRAME_GROUPS.items() if v == group)


def build_table(filename_params_pairs: Iterable[Tuple[str, Parameters]]) -> pd.DataFrame:
    row_maps = []
    for fn, params in filename_params_pairs:
        par_idx = ParamIndex.unsafe_from_config_filename(fn)
        row = {"param_index": par_idx.value, "filename": fn}
        for k in Parameters._config_keys:
            rawval = getattr(params, k)
            if k == "frames":
                value = get_frame_group(rawval)
            elif k == "method":
                value = rawval.value
            else:
                value = rawval
            row[k] = value
        row_maps.append(row)
    return pd.DataFrame(sorted(row_maps, key=lambda data: data["param_index"]))




mytab = build_table(param_by_config.items())




def count_spots(fp: Path) -> int:
    with open(fp, 'r') as fh:
        return sum(1 for _ in fh) - 1



def get_spot_counts(folder: Path, fns: Iterable[str]) -> List[Union[int, pd.NA]]:
    result = []
    for fn in fns:
        fp = folder / fn
        try:
            n_spots = count_spots(fp)
        except FileNotFoundError:
            n_spots = pd.NA
        print(f"{fn}: {n_spots}")
        result.append(n_spots)
    return result



def add_spot_counts_to_table(df: pd.DataFrame, folder: Path, param_index_column: str = "param_index") -> pd.DataFrame:
    filenames = map(lambda i: ParamIndex(i).to_roi_filename(), df[param_index_column])
    df["spot_count"] = get_spot_counts(folder=folder, fns=filenames)
    return df




mytab2 = add_spot_counts_to_table(df=mytab, folder=my_gridsearch_folder.path)
