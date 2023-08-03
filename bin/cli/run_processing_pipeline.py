"""Simple end-to-end processing pipeline for chromatin tracing data"""

import argparse
import logging
import sys
from typing import *

from gertils.pathtools import ExtantFile, ExtantFolder
import pypiper

from extract_exp_psf import workflow as run_psf_extraction
from decon import workflow as run_deconvolution
from nuc_label import workflow as run_nuclei_detection
from drift_correct import workflow as run_drift_correction
from drift_correct_accuracy_analysis import workflow as run_drift_correction_analysis, run_visualisation as run_drift_correction_accuracy_visualisation
from detect_spots import workflow as run_spot_detection
from assign_spots_to_nucs import workflow as run_spot_filtration
from cluster_analysis_cleanup import workflow as run_cleanup
from extract_spots_table import workflow as run_spot_bounding
from extract_spots import workflow as run_spot_extraction
from tracing import workflow as run_chromatin_tracing

logger = logging.getLogger(__name__)

NO_TEE_LOGS_OPTNAME = "--do-not-tee-logs"
PIPE_NAME = "looptrace"


class LooptracePipeline(pypiper.Pipeline):

    def __init__(self, config_file: ExtantFile, images_folder: ExtantFolder, output_folder: ExtantFolder, **pl_mgr_kwargs: Any) -> None:
        self.config_file = config_file
        self.images_folder = images_folder
        super(LooptracePipeline, self).__init__(name=PIPE_NAME, outfolder=str(output_folder.path), **pl_mgr_kwargs)

    def stages(self) -> List[Callable]:
        conf_data_pair = (self.config_file, self.images_folder)
        conf_only = (self.config_file, )
        func_args_pairs = (
            ("psf_extraction", run_psf_extraction, conf_data_pair),
            ("deconvolution", run_deconvolution, conf_data_pair), 
            ("nuclei_detection", run_nuclei_detection, conf_data_pair), 
            ("drift_correction", run_drift_correction, conf_data_pair), 
            ("drift_correction_accuracy_analysis", run_drift_correction_analysis, conf_data_pair), 
            ("drift_correction_accuracy_visualisation", run_drift_correction_accuracy_visualisation, conf_only), 
            ("spot_detection", run_spot_detection, conf_data_pair),
            ("spot_filtration", run_spot_filtration, conf_data_pair), 
            ("clean_1", run_cleanup, conf_only),
            ("spot_bounding", run_spot_bounding, conf_data_pair),
            ("spot_extraction", run_spot_extraction, conf_data_pair),
            ("clean_2", run_cleanup, conf_only), 
            ("tracing", run_chromatin_tracing, conf_data_pair),
            ("clean_3", run_cleanup, conf_only),
        )
        return [pypiper.Stage(func=fxn, f_args=fxn_args, name=name) for name, fxn, fxn_args in func_args_pairs]


def parse_cli(args: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A pipeline to process microscopy imaging data to trace chromatin fiber with FISH probes")
    parser.add_argument("-C", "--config-file", type=ExtantFile.from_string, required=True, help="Path to the main configuration file")
    parser.add_argument("-I", "--images-folder", type=ExtantFolder.from_string, required=True, help="Path to the root folder with imaging data to process")
    parser.add_argument("-O", "--output-folder", type=ExtantFolder.from_string, required=True, help="Path to folder for pypiper output")
    parser.add_argument(NO_TEE_LOGS_OPTNAME, action="store_true", help="Do not tee logging output from pypiper manager")
    parser = pypiper.add_pypiper_args(
        parser, 
        groups=("pypiper", "checkpoint"),
        args=("start-point", ),
        )
    return parser.parse_args(args)


def init(opts: argparse.Namespace) -> LooptracePipeline:
    kwargs = {"config_file": opts.config_file, "images_folder": opts.images_folder, "output_folder": opts.output_folder}
    if opts.do_not_tee_logs:
        kwargs["multi"] = True
    logger.info(f"Building {PIPE_NAME} pipeline from {opts.config_file.path}, to use images from {opts.images_folder.path}")
    return LooptracePipeline(**kwargs)


def main(cmdl):
    opts = parse_cli(cmdl)
    pipeline = init(opts)
    logger.info("Running pipeline")
    pipeline.run(start_point=opts.start_point, stop_after=opts.stop_after)
    pipeline.wrapup()


if __name__ == "__main__":
    main(sys.argv[1:])
