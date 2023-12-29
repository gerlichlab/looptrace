"""Simple end-to-end processing pipeline for chromatin tracing data"""

import argparse
import logging
import sys
from typing import *

from gertils import ExtantFile, ExtantFolder
import pypiper

from pipeline_precheck import workflow as pretest
from convert_datasets_to_zarr import one_to_one as run_zarr_production
from extract_exp_psf import workflow as run_psf_extraction
from run_bead_roi_generation import workflow as gen_all_bead_rois
from run_bead_roi_partition import workflow as partition_bead_rois
from analyse_detected_bead_rois import workflow as run_all_bead_roi_detection_analysis
from decon import workflow as run_deconvolution
#from nuc_label import workflow as run_nuclei_detection
from looptrace.Drifter import coarse_correction_workflow as run_coarse_drift_correction, fine_correction_workflow as run_fine_drift_correction
from drift_correct_accuracy_analysis import workflow as run_drift_correction_analysis, run_visualisation as run_drift_correction_accuracy_visualisation
from detect_spots import workflow as run_spot_detection
from run_spot_proximity_filtration import workflow as run_spot_proximity_filtration
from assign_spots_to_nucs import workflow as run_spot_nucleus_filtration
from cluster_analysis_cleanup import workflow as run_cleanup
from extract_spots_table import workflow as run_spot_bounding
from extract_spots import workflow as run_spot_extraction
from extract_spots_cluster_cleanup import workflow as run_spot_zipping
from tracing import workflow as run_chromatin_tracing
from looptrace.Tracer import run_frame_name_and_distance_application
from run_tracing_qc import workflow as qc_label_and_filter_traces


logger = logging.getLogger(__name__)

DECON_STAGE_NAME = "deconvolution"
NO_TEE_LOGS_OPTNAME = "--do-not-tee-logs"
PIPE_NAME = "looptrace"
SPOT_DETECTION_STAGE_NAME = "spot_detection"
TRACING_QC_STAGE_NAME = "tracing_QC"


class LooptracePipeline(pypiper.Pipeline):
    """Main looptrace processing pipeline"""

    def __init__(self, config_file: ExtantFile, images_folder: ExtantFolder, output_folder: ExtantFolder, **pl_mgr_kwargs: Any) -> None:
        self.config_file = config_file
        self.images_folder = images_folder
        super(LooptracePipeline, self).__init__(name=PIPE_NAME, outfolder=str(output_folder.path), **pl_mgr_kwargs)

    @staticmethod
    def name_fun_getargs_bundles() -> List[Tuple[str, callable, Callable[[Tuple[ExtantFile, ExtantFolder]], Union[Tuple[ExtantFile], Tuple[ExtantFile, ExtantFolder]]]]]:
        take1 = lambda config, _: (config, )
        take2 = lambda config, images: (config, images)
        return [
            ("pipeline_precheck", pretest, take1),
            ("zarr_production", run_zarr_production, take2),
            ("psf_extraction", run_psf_extraction, take2),
            (DECON_STAGE_NAME, run_deconvolution, take2), # Really just for denoising, no need for structural disambiguation
            ("drift_correction__coarse", run_coarse_drift_correction, take2), 
            #("nuclei_detection", run_nuclei_detection, take2), 
            ("bead_roi_generation", gen_all_bead_rois, take2), # Find/define all the bead ROIs in each (FOV, frame) pair.
            # Count detected bead ROIs for each timepoint, mainly to see if anything went awry during some phase of the imaging, e.g. air bubble.
            ("bead_roi_detection_analysis", run_all_bead_roi_detection_analysis, take2),
            ("bead_roi_partition", partition_bead_rois, take2),
            ("drift_correction__fine", run_fine_drift_correction, take2),
            ("drift_correction_accuracy_analysis", run_drift_correction_analysis, take2), 
            ("drift_correction_accuracy_visualisation", run_drift_correction_accuracy_visualisation, take1), 
            (SPOT_DETECTION_STAGE_NAME, run_spot_detection, take2), # generates *_rois.csv (regional spots)
            ("spot_proximity_filtration", run_spot_proximity_filtration, take2),
            ("spot_nucleus_filtration", run_spot_nucleus_filtration, take2), 
            ("clean_1", run_cleanup, take1),
            ("spot_bounding", run_spot_bounding, take2), # computes pad_x_min, etc.; writes *_dc_rois.csv (much bigger, since regional spots x frames)
            ("spot_extraction", run_spot_extraction, take2),
            ("spot_zipping", run_spot_zipping, take2),
            ("clean_2", run_cleanup, take1), 
            ("tracing", run_chromatin_tracing, take2),
            ("spot_region_distances", run_frame_name_and_distance_application, take2), 
            (TRACING_QC_STAGE_NAME, qc_label_and_filter_traces, take2),
            ("clean_3", run_cleanup, take1),
        ]

    def stages(self) -> List[Callable]:
        return [
            pypiper.Stage(func=fxn, f_args=get_args(self.config_file, self.images_folder), name=name) 
            for name, fxn, get_args in self.name_fun_getargs_bundles()
        ]


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
