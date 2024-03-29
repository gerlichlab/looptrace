"""Run spot detection over a grid of parameterizations."""

import argparse
from dataclasses import dataclass
import itertools
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import *

from gertils import ExtantFile, ExtantFolder, NonExtantPath
from detect_spots import ParamPatch, Parameters, workflow as run_spot_detection
from looptrace.SpotPicker import DetectionMethod, CROSSTALK_SUBTRACTION_KEY

__author__ = ["Vince Reuter"]


CONFIG_PREFIX = "config"
ROI_PREFIX = "rois"
CONFIG_FILETYPE = "json"
ROI_FILETYPE = "csv"

logger = logging.getLogger(__name__)


@dataclass
class FrameSet:
    frames: List[int]


@dataclass
class MethodAndIntensities:
    method: DetectionMethod
    intensities: List[int]


def read_params_file(params_file: ExtantFile) -> Iterable[Parameters]:
    with open(params_file.path, 'r') as fh:
        params_config = json.load(fh)
    frame_sets = params_config["frame_sets"]
    method_intensity_pairs = [(method_name, intensity) for method_name, intensities in params_config["intensities_by_method"].items() for intensity in intensities]
    downsamplings = params_config["downsamplings"]
    only_in_nuclei = params_config["only_in_nuclei"]
    subtract_crosstalk = params_config[CROSSTALK_SUBTRACTION_KEY]
    combos = itertools.product(frame_sets, method_intensity_pairs, downsamplings, only_in_nuclei, subtract_crosstalk)
    params = []
    for frames, (method_name, intensity), downsampling, only_nucs, minus_xtalk in combos:
        method = DetectionMethod.parse(method_name)
        par = Parameters(
            frames=frames, 
            method=method, 
            threshold=intensity, 
            downsampling=downsampling, 
            only_in_nuclei=only_nucs, 
            subtract_crosstalk=minus_xtalk, 
            minimum_spot_separation=5
            )
        params.append(par)
    return params


def run_one_detection(
    params: ParamPatch, 
    rounds_config: ExtantFile, 
    params_config: ExtantFile, 
    images_folder: ExtantFolder, 
    output_file: Union[NonExtantPath, Path], 
    updated_config_file: Optional[NonExtantPath],
    ):
    return run_spot_detection(
        rounds_config=rounds_config,
        params_config=params_config, 
        images_folder=images_folder, 
        image_save_path=None, 
        params_update=params, 
        outfile=str(output_file.path if isinstance(output_file, NonExtantPath) else output_file), 
        write_config_path=None if updated_config_file is None else str(updated_config_file.path), 
        )


def workflow_new_configs(
    rounds_config: ExtantFile, 
    params_config: ExtantFile, 
    images_folder: ExtantFolder, 
    params_file: ExtantFile, 
    output_folder: ExtantFolder, 
    cores: Optional[int] = None,
    ) -> Iterable[str]:
    """Run the gridsearch from a collection of parameters for which looptrace config files haven't yet been generated."""
    parameterizations = read_params_file(params_file=params_file)
    print(f"Parameterizations count: {len(parameterizations)}")
    cores = cores or 1
    argument_bundles = []
    for param_index, params in enumerate(parameterizations):
        idx = ParamIndex(param_index)
        with open(f"params.{idx.value}.{CONFIG_FILETYPE}", 'w') as fh:
            json.dump(params.to_dict_for_json(), fh, indent=4)
        args = (
            params, 
            rounds_config,
            params_config,
            images_folder, 
            NonExtantPath(output_folder.path / idx.to_roi_filename()), 
            NonExtantPath(output_folder.path / idx.to_config_filename()), 
            )
        argument_bundles.append(args)
    return execute(argument_bundles=argument_bundles, cores=cores)


def workflow_preexisting_configs(
    rounds_config: ExtantFile, 
    params_config: ExtantFile, 
    images_folder: ExtantFolder, 
    params_outfile_pairs: Iterable[Tuple[ParamPatch, NonExtantPath]], 
    cores: Optional[int] = None,
    ) -> Iterable[str]:
    """Run the gridsearch from a collection of pre-existing, pre-generated config files ready to run looptrace."""
    print(f"Parameterizations count: {len(params_outfile_pairs)}")
    cores = cores or 1
    argument_bundles = []
    for params, outfile in params_outfile_pairs:
        args = (
            params, 
            rounds_config,
            params_config,
            images_folder, 
            outfile, 
            None,
            )
        argument_bundles.append(args)
    return execute(argument_bundles=argument_bundles, cores=cores)


def execute(argument_bundles, cores) -> Iterable[str]:
    if cores == 1:
        print("Using a single core")    
        outfiles = [run_one_detection(*args) for args in argument_bundles]
    else:
        print(f"Using {cores} cores")
        with mp.Pool(cores) as work_pool:
            outfiles = work_pool.starmap(func=run_one_detection, iterable=argument_bundles)
    return outfiles


@dataclass
class ParamIndex:
    value: int

    def __post_init__(self) -> None:
        if not isinstance(self.value, int):
            raise TypeError(f"Parameter index must be integer, not {type(self.value).__name__}")
        if self.value < 0:
            raise ValueError(f"Illegal index (must be nonnegative): {self.value}")
    
    def __eq__(self, other):
        return type(self) == type(other) and self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError(f"Cannot compare {type(self).__name__} and {type(other).__name__}")
        return self.value < other.value

    def __lteq__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        return not self.__lteq__(other)

    def __gteq__(self, other):
        return not self.__lt__(other)

    def __str__(self):
        return str(self.value)

    def to_config_filename(self) -> str:
        return f"{CONFIG_PREFIX}.{self.value}.{CONFIG_FILETYPE}"

    def to_roi_filename(self) -> str:
        return f"{ROI_PREFIX}.{self.value}.{ROI_FILETYPE}"
    
    @classmethod
    def _unsafe_from_filename(cls, fn: str, exp_prefix: str, exp_filetype: str, filetype_meaning: str) -> "ParamIndex":
        prefix, idx, filetype = fn.split(".")
        if prefix == exp_prefix and filetype == exp_filetype:
            return cls(int(idx))
        raise ValueError(f"Failed to parse parameter index from alleged {filetype_meaning} filename: {fn}")
    
    @classmethod
    def unsafe_from_config_filename(cls, fn: str) -> "ParamIndex":
        return cls._unsafe_from_filename(fn=fn, exp_prefix=CONFIG_PREFIX, exp_filetype=CONFIG_FILETYPE, filetype_meaning="config")

    @classmethod
    def unsafe_from_roi_filename(cls, fn: str) -> "ParamIndex":
        return cls._unsafe_from_filename(fn=fn, exp_prefix=ROI_PREFIX, exp_filetype=ROI_FILETYPE, filetype_meaning="ROI")


@dataclass
class IndexRange:
    lo: int
    hi: int

    def __post_init__(self):
        if not isinstance(self.lo, int) or not isinstance(self.hi, int):
            raise TypeError(f"Lower and upper limit must both be integers. Got {type(self.lo).__name__} and {type(self.hi).__name__}")

    @classmethod
    def from_string(cls, s: str) -> "IndexRange":
        lo, hi = s.split("-")
        return cls(int(lo), int(hi))
    
    def __iter__(self):
        return map(ParamIndex, range(self.lo, self.hi + 1))


def parse_cmdl(cmdl: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run spot detection over a range of parameters.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Base CLI for this project
    parser.add_argument("rounds_config", type=ExtantFile.from_string, help="Imaging rounds config file path")
    parser.add_argument("params_config", type=ExtantFile.from_string, help="Looptrace parameters config file path")
    parser.add_argument("images_folder", type=ExtantFolder.from_string, help="Path to folder with images to read.")
    
    # Specifics for gridsearch
    parser.add_argument("--parameters-grid-file", type=ExtantFile.from_string, help="Path to file which declares the gridsearch parameters")
    parser.add_argument("-O", "--output-folder", type=Path, required=True, help="Path to root folder for output")
    
    # Control flow customisation
    parser.add_argument("--cores", type=int, default=1, help="Number of processing cores to use")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results file(s)")
    parser.add_argument("--index-range", type=IndexRange.from_string, nargs="*", help="Ranges of parameter indices to use, only when running through written configs code path")

    return parser.parse_args(cmdl)


def main(cmdl: List[str]) -> None:
    opts = parse_cmdl(cmdl)
    
    if opts.parameters_grid_file is None:
        if not opts.output_folder.exists():
            raise FileNotFoundError(f"Output folder, to search for configs, doesn't exist: {opts.output_folder}")
        configs = set()
        results = set()
        for fn in os.listdir(opts.output_folder):
            try:
                configs.add(ParamIndex.unsafe_from_config_filename(fn))
            except Exception:
                try:
                    results.add(ParamIndex.unsafe_from_roi_filename(fn))
                except Exception:
                    continue
        print(f"Found {len(configs)} config file(s)")
        print(f"Found {len(results)} result file(s)")
        param_indices_to_use = configs - (set() if opts.overwrite else results)
        print(f"Preliminary indices to use: {', '.join(map(str, sorted(param_indices_to_use)))}")
        if opts.index_range:
            include_filter = set(index for indices in opts.index_range for index in indices)
            print(f"Including only from among these indices: {', '.join(map(str, sorted(include_filter)))}")
            param_indices_to_use = param_indices_to_use & include_filter
        print(f"Count of parameterisation to use: {len(param_indices_to_use)}")
        params_outfile_pairs = []
        finalize_result_path = (lambda p: p) if opts.overwrite else (lambda p: NonExtantPath(p).path)
        for idx in param_indices_to_use:
            conf_path = opts.output_folder / idx.to_config_filename()
            result_path = finalize_result_path(opts.output_folder / idx.to_roi_filename())
            print(f"Reading config: {conf_path}")
            with open(conf_path, 'r') as fh:
                conf_data = json.load(fh)
            params_outfile_pairs.append((conf_data, result_path))
        workflow_preexisting_configs(
            rounds_config=opts.rounds_config,
            params_config=opts.params_config, 
            images_folder=opts.images_folder, 
            params_outfile_pairs=params_outfile_pairs, cores=opts.cores,
            )
    else:
        if opts.output_folder.exists():
            raise FileExistsError(f"Output folder already exists: {opts.output_folder}")
        print(f"Establishing output folder: {opts.output_folder}")
        os.makedirs(opts.output_folder, exist_ok=False)
        print(f"Starting spot detection gridsearch, based on parameters file: {opts.parameters_grid_file}")
        workflow_new_configs(
            rounds_config=opts.rounds_config,
            params_config=opts.params_config, 
            images_folder=opts.images_folder, 
            params_file=opts.parameters_grid_file, 
            output_folder=ExtantFolder(opts.output_folder), 
            cores=opts.cores
        )


if __name__ == "__main__":
    main(sys.argv[1:])
