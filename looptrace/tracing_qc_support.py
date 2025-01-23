"""Supporting functions for the quality control of chromatin fiber traces"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpydoc_decorator import doc
import pandas as pd

from looptrace import FIELD_OF_VIEW_COLUMN


# Columns which uniquely identify the record of reference for intra-region distance. 
# Specifically, for any traced point, it will be have come from exactly one regional 
# timepoint, within a particular trace, within a particular field of view. 
# When trace ID is unique globally (for the whole experiment, rather than just within 
# each FOV), then the field of view's inclusion in this composite key is redundant. 
# But it's retained for greater robustness to a change in the scheme of designation 
# of trace IDs (e.g., if they were to be rest for each FOV rather than unique across 
# all FOVs).
REGION_KEY_COLUMNS = [FIELD_OF_VIEW_COLUMN, "traceId", "ref_timepoint"]


@doc(
    summary="Name the timepoints, and compute distance from each locus spot its regional spot",
    parameters=dict(
        traces_file="The path to the file to annotate with metadata and distance information", 
        timepoint_names="The sequence of timepoint names, 0-based index implying the mapping Int -> String",
    ),
    returns="The frame parsed from the given file, annotated with metadata and spatial information",
)
def apply_timepoint_names_and_spatial_information(traces_file: Path, timepoint_names: Iterable[str]) -> pd.DataFrame:
    timepoint_names: list[str] = list(timepoint_names)
    print(f"{len(timepoint_names)} timepoint names: {', '.join(timepoint_names)}")
    print(f"Reading traces: {traces_file}")
    traces = pd.read_csv(traces_file, index_col=False)
    timepoints = traces["timepoint"]
    print(f"Applying timepoint names to traces...")
    traces["timepoint_name"] = timepoints.apply(lambda t: timepoint_names[t])
    if "ref_dist" not in traces.columns:
        traces["ref_dist"] = compute_ref_timepoint_spatial_information(traces)
    return traces


@doc(
    summary="Annotate the given frame with distance between each record and its corresponding regional spot",
    parameters=dict(df="The frame to annotate with spatial information"),
    returns="The given frame, with an additional column storing distance between each spot and its corresponding regional center",
)
def compute_ref_timepoint_spatial_information(df: pd.DataFrame) -> pd.DataFrame:
    """Populate table with coordinates of probe's reference point and distance of each spot to its reference."""
    refs = df[df["timepoint"] == df["ref_timepoint"]]
    refkeys = refs[REGION_KEY_COLUMNS].apply(lambda r: tuple(map(lambda c: r[c], REGION_KEY_COLUMNS)), axis=1)
    for dim in ["z", "y", "x"]:
        refs_map = dict(zip(refkeys, refs[dim]))
        df[dim + "_ref"] = df[REGION_KEY_COLUMNS].apply(lambda r: refs_map[tuple(map(lambda c: r[c], REGION_KEY_COLUMNS))], axis=1)
    return np.sqrt((df["z_ref"] - df["z"])**2 + (df["y_ref"] - df["y"])**2 + (df["x_ref"] - df["x"])**2)
