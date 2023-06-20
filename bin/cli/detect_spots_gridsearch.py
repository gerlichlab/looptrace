"""Run spot detection over a grid of parameterizations."""

import argparse
import itertools
import copy
from dataclasses import dataclass
import json
import os
import sys
from typing import *

from looptrace.pathtools import ExtantFile, ExtantFolder, NonExtantPath
from .detect_spots import Method, Parameters, workflow as run_spot_detection

__author__ = ["Vince Reuter"]


FRAME_SETS = [(22, 23, 24, 25), (2, 8, 11, 13)]
METHOD_INTENSITY_PAIRS = [(m, i) for m, intensities in [(Method.INTENSITY, [400, 200, 100]), (Method.DIFFERENCE_OF_GAUSSIANS, [5, 4, 3, 2, 1])] for i in intensities]
DOWNSAMPLE = [2, 1]
ONLY_IN_NUCLEI = [False, True]
SUBTRACT_CROSSTALK = [False, True]


@dataclass
class FrameSet:
    frames: List[str]


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
    return parser.parse_args(cmdl)


def workflow(config_file: ExtantFile, images_folder: ExtantFolder, params_file: ExtantFile, output_folder: ExtantFolder):
    parameterizations = read_params_file(params_file=params_file)
    for param_index, params in enumerate(parameterizations):
        output_path = output_folder.path / f"rois.{param_index}.csv"
        new_config_path = output_folder.path / f"config.{param_index}.json"
        run_spot_detection(
            config_file=config_file, 
            images_folder=images_folder, 
            image_save_path=None, 
            params_update=params, 
            outfile=str(output_path), 
            write_config_path=str(new_config_path), 
            )


def main(cmdl: List[str]) -> None:
    opts = parse_cmdl(cmdl)
    os.makedirs(opts.output_folder, exist_ok=False)
    workflow(params_file=opts.parameters_grid_file, output_folder=ExtantFolder(opts.output_folder))


if __name__ == "__main__":
    main(sys.argv[1:])
