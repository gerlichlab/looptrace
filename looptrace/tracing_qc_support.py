"""Supporting functions for the quality control of chromatin fiber traces"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


REGION_KEY_COLUMNS = ["fieldOfView", "traceId", "ref_timepoint"]


def apply_timepoint_names_and_spatial_information(traces_file: Path, timepoint_names: Iterable[str]) -> pd.DataFrame:
    traces = read_traces_and_apply_timepoint_names(traces_file=traces_file, timepoint_names=timepoint_names)
    if "ref_dist" not in traces.columns:
        traces["ref_dist"] = compute_ref_timepoint_spatial_information(traces)
    return traces


def compute_ref_timepoint_spatial_information(df: pd.DataFrame) -> pd.DataFrame:
    """Populate table with coordinates of probe's reference point and distance of each spot to its reference."""
    refs = df[df["timepoint"] == df["ref_timepoint"]]
    refkeys = refs[REGION_KEY_COLUMNS].apply(lambda r: tuple(map(lambda c: r[c], REGION_KEY_COLUMNS)), axis=1)
    for dim in ["z", "y", "x"]:
        refs_map = dict(zip(refkeys, refs[dim]))
        df[dim + "_ref"] = df[REGION_KEY_COLUMNS].apply(lambda r: refs_map[tuple(map(lambda c: r[c], REGION_KEY_COLUMNS))], axis=1)
    return np.sqrt((df["z_ref"] - df["z"])**2 + (df["y_ref"] - df["y"])**2 + (df["x_ref"] - df["x"])**2)


def read_traces_and_apply_timepoint_names(traces_file: Path, timepoint_names: Iterable[str]) -> pd.DataFrame:
    timepoint_names: list[str] = list(timepoint_names)
    print(f"{len(timepoint_names)} timepoint names: {', '.join(timepoint_names)}")
    print(f"Reading traces: {traces_file}")
    traces = pd.read_csv(traces_file, index_col=False)
    timepoints = traces["timepoint"]
    print(f"Applying timepoint names to traces...")
    traces["timepoint_name"] = timepoints.apply(lambda t: timepoint_names[t])
    return traces
