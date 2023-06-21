"""Run spot detection over a grid of parameterizations."""

import argparse
from dataclasses import dataclass
import itertools
import json
import multiprocessing as mp
import os
import sys
from typing import *

import logmuse

from looptrace.pathtools import ExtantFile, ExtantFolder, NonExtantPath
from .detect_spots import Method, Parameters, workflow as run_spot_detection

__author__ = ["Vince Reuter"]


logger = None # to be built and configured during main control flow


@dataclass
class FrameSet:
    frames: List[int]


@dataclass
class MethodAndIntensities:
    method: Method
    intensities: List[int]


def read_params_file(params_file: ExtantFile) -> Iterable[Parameters]:
    with open(params_file.path, 'r') as fh:
        params_config = json.load(fh)
    frame_sets = params_config["frame_sets"]
    method_intensity_pairs = [(method_name, intensity) for method_name, intensities in params_config["intensities_by_method"].items() for intensity in intensities]
    downsamplings = params_config["downsamplings"]
    only_in_nuclei = params_config["only_in_nuclei"]
    subtract_crosstalk = params_config["subtract_crosstalk"]
    combos = itertools.product(frame_sets, method_intensity_pairs, downsamplings, only_in_nuclei, subtract_crosstalk)
    params = []
    for frames, (method_name, intensity), downsampling, only_nucs, minus_xtalk in combos:
        method = Method.parse(method_name)
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


def parse_cmdl(cmdl: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run spot detection over a range of parameters.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config_path", type=ExtantFile, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder, help="Path to folder with images to read.")
    parser.add_argument("--parameters-grid-file", required=True, type=ExtantFile, help="Path to file which declares the gridsearch parameters")
    parser.add_argument("-O", "--output-folder", type=NonExtantPath, required=True, help="Path to root folder for output")
    parser.add_argument("--cores", type=int, default=1, help="Number of processing cores to use")
    parser = logmuse.add_logging_options(parser)
    return parser.parse_args(cmdl)


def run_one_detection(params: Parameters, config_file: ExtantFile, images_folder: ExtantFolder, output_file: NonExtantPath, updated_config_file: NonExtantPath):
    return run_spot_detection(
        config_file=config_file, 
        images_folder=images_folder, 
        image_save_path=None, 
        params_update=params, 
        outfile=str(output_file.path), 
        write_config_path=str(updated_config_file.path), 
        )



def workflow(config_file: ExtantFile, images_folder: ExtantFolder, params_file: ExtantFile, output_folder: ExtantFolder, cores: Optional[int] = None):
    parameterizations = read_params_file(params_file=params_file)
    logger.info(f"Parameterizations count: {len(parameterizations)}")
    cores = cores or 1
    argument_bundles = ((
        params, 
        config_file, 
        images_folder, 
        NonExtantPath(output_folder.path / f"rois.{param_index}.csv"), 
        NonExtantPath(output_folder.path / f"config.{param_index}.json")
        )  for param_index, params in enumerate(parameterizations))
    if cores == 1:
        logger.info("Using a single core")
        for args in argument_bundles:
            run_one_detection(*args)
    else:
        logger.info(f"Using {cores} cores")
        with mp.Pool(cores) as work_pool:
            work_pool.starmap_async(func=run_one_detection, iterable=argument_bundles)


def main(cmdl: List[str]) -> None:
    opts = parse_cmdl(cmdl)
    
    global logger
    logger = logmuse.logger_from_cli(opts)

    logger.debug(f"Establishing output folder: {opts.output_folder}")
    os.makedirs(opts.output_folder, exist_ok=False)

    logger.info(f"Starting spot detection gridsearch, based on parameters file: {opts.parameters_grid_file}")
    workflow(params_file=opts.parameters_grid_file, output_folder=ExtantFolder(opts.output_folder))


if __name__ == "__main__":
    main(sys.argv[1:])
