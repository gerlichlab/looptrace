"""Supporting functions for the quality control of chromatin fiber traces"""

from pathlib import Path
from typing import *

import numpy as np
import pandas as pd

from looptrace import read_table_pandas


def apply_frame_names_and_spatial_information(traces_file: Path, frame_names: List[str]) -> pd.DataFrame:
    traces = read_traces_and_apply_frame_names(traces_file=traces_file, frame_names=frame_names)
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


def read_traces_and_apply_frame_names(traces_file: Path, frame_names: List[str]) -> pd.DataFrame:
    print(f"{len(frame_names)} frame names: {', '.join(frame_names)}")
    print(f"Reading traces: {traces_file}")
    traces = read_table_pandas(traces_file)
    timepoints = traces["frame"]
    if timepoints.nunique() != len(frame_names):
        raise ValueError(f"Traces table has {timepoints.nunique()} unique timepoints, but there are {len(frame_names)} frame names given!")
    print(f"Applying frame names to traces...")
    traces["frame_name"] = timepoints.apply(lambda t: frame_names[t])
    return traces
