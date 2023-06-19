"""Simple end-to-end processing pipeline for chromatin tracing data"""

import argparse
import sys
from typing import *

from pypiper import Pipeline, Stage

from looptrace.ImageHandler import ImageHandler
from looptrace.pathtools import ExtantFile, ExtantFolder

from .extract_exp_psf import workflow as run_point_spread_function_generation
from .decon import workflow as run_deconvolution
from .nuc_label import workflow as run_nuclei_detection
from .drift_correct import workflow as run_drift_correction
from .detect_spots import workflow as run_spot_detection
from .cluster_analysis_cleanup import workflow as run_cleanup
from .extract_spots_table import workflow as run_spot_bounding
from .extract_spots import workflow as run_spot_extraction
from .tracing import workflow as run_chromatin_tracing


class LooptracePipeline(Pipeline):

    def __init__(self, config_file: ExtantFile, images_folder: ExtantFolder) -> None:
        self.config_file = config_file
        self.images_folder = images_folder

    def stages(self) -> List[Callable]:
        conf_data_pair = (self.config_file, self.images_folder)
        conf_only = (self.config_file, )
        func_args_pairs = (
            (run_point_spread_function_generation, conf_data_pair),
            (run_deconvolution, conf_data_pair), 
            (run_nuclei_detection, conf_data_pair), 
            (run_drift_correction, conf_data_pair), 
            (run_spot_detection, conf_data_pair),
            (run_cleanup, conf_only),
            (run_spot_bounding, conf_data_pair),
            (run_spot_extraction, conf_data_pair),
            (run_cleanup, conf_only), 
            (run_chromatin_tracing, conf_data_pair),
            (run_cleanup, conf_only),
        )
        return [Stage(func=fxn, f_args=fxn_args) for fxn, fxn_args in func_args_pairs]


def parse_cmdl(cmdl: argparse.Namespace):
    parser = argparse.ArgumentParser(description="A pipeline to process microscopy imaging data to trace chromatin fiber with FISH probes")
    parser.add_argument("-C", "--config-file", type=ExtantFile, required=True, help="Path to the main configuration file")
    parser.add_argument("-I", "--images-folder", type=ExtantFolder, required=True, help="Path to the root folder with imaging data to process")
    return parser.parse_args(cmdl)


def main(cmdl):
    opts = parse_cmdl(cmdl)
    image_handler = ImageHandler(config_path=str(opts.config_file.path), image_path=str(opts.images_folder.path))
    pipeline = LooptracePipeline(image_handler=image_handler)
    pipeline.run()


if __name__ == "__main__":
    main(sys.argv[1:])
