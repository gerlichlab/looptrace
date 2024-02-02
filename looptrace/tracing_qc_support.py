"""Supporting functions for the quality control of chromatin fiber traces"""

import dataclasses
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
import yaml

from looptrace import *
from looptrace.numeric_types import NumberLike as Numeric


@dataclasses.dataclass
class TracingQCParameters:
    """Encapsulate the QC parameters for filtering trace points."""
    min_signal_noise_ratio: Numeric
    max_sd_xy_nanometers: Numeric
    max_sd_z_nanometers: Numeric
    max_dist_from_region_center_nanometers: Numeric

    @classmethod
    def _build(cls, data: Dict[str, Numeric], keys: Iterable[Tuple[str, str]]) -> "TracingQCParameters":
        missing = []
        params = {}
        for int_key, ext_key in keys:
            try:
                params[int_key] = data[ext_key]
            except KeyError:
                missing.append(ext_key)
        if missing:
            raise Exception(f"Missing {len(missing)} key(s) for QC parameters data: {', '.join(missing)}")
        return cls(**params)

    @classmethod
    def from_dict_external(cls, data: Dict[str, Numeric]) -> "TracingQCParameters":
        return cls._build(data=data, keys=cls._internal_to_external().items())
    
    @classmethod
    def from_dict_internal(cls, data: Dict[str, Numeric]) -> "TracingQCParameters":
        return cls._build(data=data, keys=((f.name, f.name) for f in dataclasses.fields(cls)))
    
    def to_dict_internal(self) -> Dict[str, Numeric]:
        return dataclasses.asdict(self)
    
    @classmethod
    def _internal_to_external(cls):
        return {
            "min_signal_noise_ratio": SIGNAL_NOISE_RATIO_NAME, 
            "max_sd_xy_nanometers": SIGMA_XY_MAX_NAME, 
            "max_sd_z_nanometers": SIGMA_Z_MAX_NAME,
            "max_dist_from_region_center_nanometers": MAX_DISTANCE_SPOT_FROM_REGION_NAME,
        }


def apply_frame_names_and_spatial_information(traces_file: Path, config_file: Path) -> pd.DataFrame:
    traces, _ = read_traces_and_apply_frame_names(traces_file=traces_file, config_file=config_file)
    if "ref_dist" not in traces.columns:
        traces["ref_dist"] = compute_ref_frame_spatial_information(traces)
    return traces


def compute_ref_frame_spatial_information(df: pd.DataFrame) -> pd.DataFrame:
    """Populate table with coordinates of probe's reference point and distance of each spot to its reference."""
    refs = df[df['frame'] == df['ref_frame']]
    keycols = ['pos_index', 'trace_id', 'ref_frame']
    refkeys = refs[keycols].apply(lambda r: tuple(map(lambda c: r[c], keycols)), axis=1)
    for dim in ['z', 'y', 'x']:
        refs_map = dict(zip(refkeys, refs[dim]))
        df[dim + '_ref'] = df[keycols].apply(lambda r: refs_map[tuple(map(lambda c: r[c], keycols))], axis=1)
    return np.sqrt((df['z_ref'] - df['z'])**2 + (df['y_ref'] - df['y'])**2 + (df['x_ref'] - df['x'])**2)


def read_traces_and_apply_frame_names(traces_file: Path, config_file: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    print(f"Reading config file: {config_file}")
    with open(config_file, 'r') as fh:
        config = yaml.safe_load(fh)
    frame_names = config["frame_name"]
    print(f"{len(frame_names)} frame names: {', '.join(frame_names)}")
    print(f"Reading traces: {traces_file}")
    traces = read_table_pandas(traces_file)
    print(f"Applying frame names to traces")
    traces["frame_name"] = traces.apply(lambda row: frame_names[row["frame"]], axis=1)
    return traces, config
