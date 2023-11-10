"""Supporting functions for the quality control of chromatin fiber traces"""

import dataclasses
import itertools
import json
from pathlib import Path
from typing import *

from gertils import ExtantFile
import numpy as np
import pandas as pd
import yaml

from looptrace.numeric_types import NumberLike as Numeric

QC_PASS_COUNT_COLUMN = "n_qc_pass"


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
            "min_signal_noise_ratio": "A_to_BG", 
            "max_sd_xy_nanometers": "sigma_xy_max", 
            "max_sd_z_nanometers": "sigma_z_max",
            "max_dist_from_region_center_nanometers": "max_dist"
        }


def add_qc_and_parse_config(traces_file: Path, config_file: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    traces, config = read_traces_and_apply_frame_names(traces_file=traces_file, config_file=config_file)
    print("Applying general trace QC")
    qc_params = TracingQCParameters.from_dict_external(config)
    print(f"QC params:\n{json.dumps(qc_params.to_dict_internal(), indent=2)}")
    traces['QC'] = tracing_qc(df=traces, parameters=qc_params)
    return traces, config


def apply_qc_filtration_and_write_results(
        traces_file: Path, 
        config_file: Path, 
        min_trace_length: Optional[int], 
        exclusions: Optional[Iterable[str]],
        ) -> Tuple[Path, Path]:
    _, traces = read_traces_parse_frame_names_and_apply_traces_qc(
        traces_file=traces_file, 
        config_file=config_file, 
        min_trace_length=min_trace_length, 
        exclusions=exclusions,
        )
    unfiltered_output_file = traces_file.with_suffix(".qc_unfiltered.csv")
    filtered_output_file = traces_file.with_suffix(".qc_filtered.csv")
    traces.to_csv(unfiltered_output_file, index=False)
    traces[traces["QC"] == 1].to_csv(filtered_output_file, index=False)
    return unfiltered_output_file, filtered_output_file


def compute_ref_frame_spatial_information(df: pd.DataFrame) -> pd.DataFrame:
    """Populate table with coordinates of probe's reference point and distance of each spot to its reference."""
    refs = df[df['frame'] == df['ref_frame']]
    keycols = ['pos_index', 'trace_id', 'ref_frame']
    refkeys = refs[keycols].apply(lambda r: tuple(map(lambda c: r[c], keycols)), axis=1)
    for dim in ['z', 'y', 'x']:
        refs_map = dict(zip(refkeys, refs[dim]))
        df[dim + '_ref'] = df[keycols].apply(lambda r: refs_map[tuple(map(lambda c: r[c], keycols))], axis=1)
    return np.sqrt((df['z_ref'] - df['z'])**2 + (df['y_ref'] - df['y'])**2 + (df['x_ref'] - df['x'])**2)


def count_tracing_qc_passes_single(traces: pd.DataFrame, parameters: TracingQCParameters) -> int:
    return tracing_qc(df=traces, parameters=parameters).sum()


def count_tracing_qc_passes_multiple(traces: pd.DataFrame, parameters: Iterable[TracingQCParameters]) -> pd.DataFrame:
    rows = []
    for pars in parameters:
        r = pars.to_dict_internal()
        r[QC_PASS_COUNT_COLUMN] = count_tracing_qc_passes_single(traces=traces, parameters=pars)
        rows.append(r)
    return pd.DataFrame(data=rows)


def count_tracing_qc_passes_from_gridmap(traces: pd.DataFrame, parameters: Mapping[str, Iterable[Numeric]]) -> pd.DataFrame:
    flattened = [[(k, v) for v in vs] for k, vs in parameters.items()]
    combos = list(itertools.product(*flattened))
    #print(combos)
    pars = (TracingQCParameters.from_dict_internal(dict(ps)) for ps in combos)
    return count_tracing_qc_passes_multiple(traces=traces, parameters=pars)


def read_traces_and_apply_frame_names(traces_file: Path, config_file: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    print(f"Reading config file: {config_file}")
    with open(config_file, 'r') as fh:
        config = yaml.safe_load(fh)
    frame_names = config["frame_name"]
    print(f"{len(frame_names)} frame names: {', '.join(frame_names)}")
    print(f"Reading traces: {traces_file}")
    traces = pd.read_csv(traces_file, index_col=0)
    print(f"Applying frame names to traces")
    traces["frame_name"] = traces.apply(lambda row: frame_names[row["frame"]], axis=1)
    return traces, config


def read_traces_parse_frame_names_and_apply_traces_qc(
        traces_file: Path, 
        config_file: Path, 
        min_trace_length: Optional[int], 
        exclusions: Optional[Iterable[str]],
        ) -> Tuple[List[str], pd.DataFrame]:
    traces, config = add_qc_and_parse_config(traces_file=traces_file, config_file=config_file)
    print("Applying trace length QC")
    filter_by_name = lambda df: df[~df["frame_name"].isin(exclusions)] if exclusions else df
    filter_by_length = lambda df: tracing_length_qc(traces=df, min_length=min_trace_length) if min_trace_length else df
    traces = filter_by_length(filter_by_name(traces))
    return config["frame_name"], traces


def tracing_length_qc(traces: pd.DataFrame, min_length: int = 0) -> pd.DataFrame:
    '''Remove traces that are not sufficiently long.'''
    return traces.groupby('trace_id').filter(lambda x: x['QC'].sum() >= min_length)


def tracing_qc(df: pd.DataFrame, parameters: TracingQCParameters) -> np.array:
    qc = np.ones((len(df)), dtype=bool)

    try:
        ref_dist = df.ref_dist
    except (AttributeError, KeyError):
        ref_dist = compute_ref_frame_spatial_information(df=df)
        df['ref_dist'] = ref_dist
    
    max_dist = parameters.max_dist_from_region_center_nanometers
    if max_dist > 0:
        qc = qc & (ref_dist < max_dist)
    
    qc = qc & (df['A'] > (parameters.min_signal_noise_ratio * df['BG']))
    qc = qc & (df['sigma_xy'] < parameters.max_sd_xy_nanometers)
    qc = qc & (df['sigma_z'] < parameters.max_sd_z_nanometers)
    qc = qc & df.apply(lambda row: row["sigma_z"] < row["z_px"] < row["spot_box_z"] - row["sigma_z"], axis=1)
    qc = qc & df.apply(lambda row: row["sigma_xy"] < row["y_px"] < row["spot_box_y"] - row["sigma_xy"], axis=1)
    qc = qc & df.apply(lambda row: row["sigma_xy"] < row["x_px"] < row["spot_box_x"] - row["sigma_xy"], axis=1)

    return qc.astype(int)


def write_tracing_qc_passes_from_gridfile(
        traces_file: ExtantFile, 
        config_file: ExtantFile, 
        gridfile: ExtantFile, 
        outfile: Path, 
        exclusions: Optional[Iterable[str]]
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    traces, _ = read_traces_and_apply_frame_names(traces_file=traces_file.path, config_file=config_file.path)
    ref_dist = compute_ref_frame_spatial_information(df=traces)
    traces["ref_dist"] = ref_dist
    traces = traces[~traces["frame_name"].isin(exclusions)] if exclusions else traces
    with open(gridfile.path, 'r') as fh:
        pargrid = json.load(fh)
    print(f"Parameters:\n{json.dumps(pargrid, indent=2)}")
    result = count_tracing_qc_passes_from_gridmap(traces=traces, parameters=pargrid)
    print(f"Saving QC pass counts: {outfile}")
    result.to_csv(outfile, index=False)
    return traces, result
