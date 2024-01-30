"""Simple end-to-end processing pipeline for chromatin tracing data"""

import argparse
from enum import Enum
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import *

from gertils import ExtantFile, ExtantFolder
import pypiper

from looptrace import *
from looptrace.Drifter import coarse_correction_workflow as run_coarse_drift_correction, fine_correction_workflow as run_fine_drift_correction
from looptrace.ImageHandler import ImageHandler
from looptrace.NucDetector import NucDetector
from looptrace.Tracer import run_frame_name_and_distance_application

from pipeline_precheck import workflow as pretest
from convert_datasets_to_zarr import one_to_one as run_zarr_production
from extract_exp_psf import workflow as run_psf_extraction
from run_bead_roi_generation import workflow as gen_all_bead_rois
from analyse_detected_bead_rois import workflow as run_all_bead_roi_detection_analysis
from decon import workflow as run_deconvolution
from nuc_label import workflow as run_nuclei_detection
from drift_correct_accuracy_analysis import workflow as run_drift_correction_analysis, run_visualisation as run_drift_correction_accuracy_visualisation
from detect_spots import workflow as run_spot_detection
from assign_spots_to_nucs import workflow as run_spot_nucleus_filtration
from extract_spots_table import workflow as run_spot_bounding
from extract_spots import workflow as run_spot_extraction
from extract_spots_cluster_cleanup import workflow as run_spot_zipping
from tracing import workflow as run_chromatin_tracing

logger = logging.getLogger(__name__)

DECON_STAGE_NAME = "deconvolution"
NO_TEE_LOGS_OPTNAME = "--do-not-tee-logs"
PIPE_NAME = "looptrace"
SPOT_DETECTION_STAGE_NAME = "spot_detection"
TRACING_QC_STAGE_NAME = "tracing_QC"


class SpotType(Enum):
    REGIONAL = "regional"
    LOCUS_SPECIFIC = "locus-specific"


def partition_bead_rois(config_file: ExtantFile, images_folder: ExtantFolder):
    """Run the bead ROIs partitioning program / pipeline step."""
    H = ImageHandler(config_path=config_file, image_path=images_folder)
    prog_path = f"{LOOPTRACE_JAVA_PACKAGE}.PartitionIndexedDriftCorrectionRois"
    cmd_parts = [
        "java", 
        "-cp",
        str(LOOPTRACE_JAR_PATH),
        prog_path, 
        "--beadRoisRoot",
        str(H.bead_rois_path), 
        "--numShifting", 
        str(H.num_bead_rois_for_drift_correction),
        "--numAccuracy",
        str(H.num_bead_rois_for_drift_correction_accuracy),
        "--outputFolder",
        str(H.bead_rois_path),
    ]
    print(f"Running bead ROI partitioning: {' '.join(cmd_parts)}")
    subprocess.check_call(cmd_parts)


def run_spot_proximity_filtration(config_file: ExtantFile, images_folder: ExtantFolder) -> None:
    H = ImageHandler(config_path=config_file, image_path=images_folder)
    min_spot_sep = H.minimum_spot_separation
    if min_spot_sep <= 0:
        print(f"No spot filtration on proximity to be done, as minimum separation = {min_spot_sep}")
        return
    
    # Detemination of the CLI flag or option/argument pair to add to specify regional grouping
    region_group_data = H.config.get("regional_spots_grouping", "NONE")
    if region_group_data == "NONE":
        groups_extras = ["--noRegionGrouping"]
    else:
        try:
            semantic = region_group_data["semantic"]
            region_groups = region_group_data["groups"]
        except (KeyError, TypeError):
            print("Could not parse regional grouping; check semantic (string) and groups (list-of-lists) are defined.")
            raise
        if semantic.lower() == "permissive":
            optname = "--proximityPermissions"
        elif semantic.lower() == "prohibitive":
            optname = "--proximityProhibitions"
        else:
            raise ValueError(f"Unrecogised semantic for regional timepoint grouping: '{semantic}'")
        argval = ",".join(",".join([f"{r}={i}" for r in rs]) for i, rs in enumerate(region_groups))
        groups_extras = [optname, argval]
    
    # Command construction, printing, and execution
    prog_path = f"{LOOPTRACE_JAVA_PACKAGE}.LabelAndFilterRois"
    cmd_parts = [
        "java", 
        "-cp",
        str(LOOPTRACE_JAR_PATH),
        prog_path, 
        "--spotsFile",
        str(H.raw_spots_file),
        "--driftFile", 
        str(H.drift_correction_file__fine),
        "--spotSeparationThresholdValue", 
        str(H.minimum_spot_separation),
        "--spotSeparationThresholdType",
        "EachAxisAND",
        "--unfilteredOutputFile",
        str(H.proximity_labeled_spots_file_path),
        "--filteredOutputFile",
        str(H.proximity_filtered_spots_file_path),
        "--handleExtantOutput",
        "OVERWRITE" # TODO: parameterise this, see: https://github.com/gerlichlab/looptrace/issues/142
    ] + groups_extras
    print(f"Running spot filtering on proximity: {' '.join(cmd_parts)}")
    subprocess.check_call(cmd_parts)


def plot_spot_counts(config_file: ExtantFile, spot_type: "SpotType") -> None:
    """Plot the unfiltered and filtered counts of spots (regional or locus-specific).
    
    Plot heatmaps stratifying spot counts by timepoint and by field-of-view, and 
    plot grouped (by timepoint) barchart with unfiltered and filtered side-by-side 
    for each timepoint, with count summed across all fields of view.

    Arguments
    ---------
    config_file : ExtantFile
        Path to looptrace processing configuration file, used to build a 
        looptrace.ImageHandlerfor pointing to the relevant filepaths
    spot_type : SpotType
        Enumeration value indicating the type of spot counts data to plot
    
    Raises
    ------
    FileNotFoundError: if the path to the analysis script to run isn't found as a file
    ValueError: if the given spot_type value isn't recognised
    """
    H = ImageHandler(config_path=config_file)
    output_folder = H.analysis_path
    analysis_script_file = Path(os.path.dirname(__file__)) / "spot_counts_visualisation.R"
    if not analysis_script_file.is_file():
        raise FileNotFoundError(f"Missing regional spot counts plot script: {analysis_script_file}")
    if spot_type == SpotType.REGIONAL:
        unfiltered = H.raw_spots_file
        filtered = H.proximity_filtered_spots_file_path
        probe_name_extra = ["--probe-names"] + H.frame_names
    elif spot_type == SpotType.LOCUS_SPECIFIC:
        unfiltered = H.traces_file_qc_unfiltered
        filtered = H.traces_file_qc_filtered
        probe_name_extra = []
    else:
        raise ValueError(f"Illegal spot_type for plotting spot counts: {spot_type}")
    cmd_parts = [
        "Rscript", 
        str(analysis_script_file), 
        "--unfiltered-spots-file", 
        str(unfiltered),
        "--filtered-spots-file",
        str(filtered),
        "--spot-file-type", 
        spot_type.value,
        "-o", 
        output_folder, 
        ] + probe_name_extra
    print(f"Running spot count plotting command: {' '.join(cmd_parts)}")
    return subprocess.check_call(cmd_parts)


def qc_label_and_filter_traces(config_file: ExtantFile, images_folder: ExtantFolder) -> None:
    H = ImageHandler(config_path=config_file, image_path=images_folder)
    prog_path = f"{LOOPTRACE_JAVA_PACKAGE}.LabelAndFilterTracesQC"
    cmd_parts = [
        "java", 
        "-cp",
        str(LOOPTRACE_JAR_PATH),
        prog_path, 
        "--tracesFile",
        str(H.traces_path_enriched),
        "--maxDistanceToRegionCenter", 
        str(H.config[MAX_DISTANCE_SPOT_FROM_REGION_NAME]),
        "--minSNR",
        str(H.config[SIGNAL_NOISE_RATIO_NAME]),
        "--maxSigmaXY",
        str(H.config[SIGMA_XY_MAX_NAME]),
        "--maxSigmaZ",
        str(H.config[SIGMA_Z_MAX_NAME]),
    ]

    exclusions = H.illegal_frames_for_trace_support
    if not exclusions:
        raise ValueError("No probes to exclude from trace support were provided!")
    cmd_parts.extend(["--exclusions", ','.join(H.illegal_frames_for_trace_support)]) # format required for parsing by scopt
    
    print(f"Running QC filtering of tracing supports: {' '.join(cmd_parts)}")
    subprocess.check_call(cmd_parts)


def compute_locus_pairwise_distances(config_file: ExtantFile) -> None:
    """Run the locus pairwise distances computation program.

    Arguments
    ---------
    config_file : ExtantFile
    """
    H = ImageHandler(config_path=config_file)
    cmd_parts = [
        "java", 
        "-cp",
        str(LOOPTRACE_JAR_PATH),
        f"{LOOPTRACE_JAVA_PACKAGE}.ComputeLocusPairwiseDistances",
        "--tracesFile",
        str(H.traces_file_qc_filtered),
        "-O", 
        H.analysis_path,
    ]
    print(f"Running distance computation command: {' '.join(cmd_parts)}")
    return subprocess.check_call(cmd_parts)


def compute_region_pairwise_distances(config_file: ExtantFile) -> None:
    """Run the regional pairwise distances computation program.

    Arguments
    ---------
    config_file : ExtantFile
    """
    H = ImageHandler(config_path=config_file)
    cmd_parts = [
        "java", 
        "-cp",
        str(LOOPTRACE_JAR_PATH),
        f"{LOOPTRACE_JAVA_PACKAGE}.ComputeRegionPairwiseDistances",
        "--roisFile",
        str(H.proximity_filtered_spots_file_path),
        "-O", 
        H.analysis_path,
    ]
    print(f"Running distance computation command: {' '.join(cmd_parts)}")
    return subprocess.check_call(cmd_parts)


def drift_correct_nuclei(config_file: ExtantFile, images_folder: ExtantFolder) -> Path:
    H = ImageHandler(config_path=config_file, image_path=images_folder)
    N = NucDetector(H)
    return N.coarse_drift_correction_workflow()


class LooptracePipeline(pypiper.Pipeline):
    """Main looptrace processing pipeline"""

    def __init__(self, config_file: ExtantFile, images_folder: ExtantFolder, output_folder: ExtantFolder, **pl_mgr_kwargs: Any) -> None:
        self.config_file = config_file
        self.images_folder = images_folder
        super(LooptracePipeline, self).__init__(name=PIPE_NAME, outfolder=str(output_folder.path), **pl_mgr_kwargs)

    @staticmethod
    def name_fun_getargs_bundles() -> List[Tuple[str, callable, Callable[[Tuple[ExtantFile, ExtantFolder]], Union[Tuple[ExtantFile], Tuple[ExtantFile, ExtantFolder]]]]]:
        take1 = lambda config_file, _: (config_file, )
        take1_with_spot_type = lambda spot_type: (lambda config_file, _2: take1(config_file, _2) + (spot_type, ))
        take2 = lambda config_file, images_folder: (config_file, images_folder)
        return [
            ("pipeline_precheck", pretest, take1),
            ("zarr_production", run_zarr_production, take2),
            ("nuclei_detection", run_nuclei_detection, take2),
            ("nuclei_drift_correction", drift_correct_nuclei, take2),
            ("psf_extraction", run_psf_extraction, take2),
            (DECON_STAGE_NAME, run_deconvolution, take2), # Really just for denoising, no need for structural disambiguation
            ("drift_correction__coarse", run_coarse_drift_correction, take2), 
            ("bead_roi_generation", gen_all_bead_rois, take2), # Find/define all the bead ROIs in each (FOV, frame) pair.
            # Count detected bead ROIs for each timepoint, mainly to see if anything went awry during some phase of the imaging, e.g. air bubble.
            ("bead_roi_detection_analysis", run_all_bead_roi_detection_analysis, take2),
            ("bead_roi_partition", partition_bead_rois, take2),
            ("drift_correction__fine", run_fine_drift_correction, take2),
            ("drift_correction_accuracy_analysis", run_drift_correction_analysis, take2), 
            ("drift_correction_accuracy_visualisation", run_drift_correction_accuracy_visualisation, take1), 
            (SPOT_DETECTION_STAGE_NAME, run_spot_detection, take2), # generates *_rois.csv (regional spots)
            ("spot_proximity_filtration", run_spot_proximity_filtration, take2),
            ("spot_counts_visualisation__regional", plot_spot_counts, take1_with_spot_type(SpotType.REGIONAL)), 
            ("spot_nucleus_filtration", run_spot_nucleus_filtration, take2), 
            ("spot_bounding", run_spot_bounding, take2), # computes pad_x_min, etc.; writes *_dc_rois.csv (much bigger, since regional spots x frames)
            ("spot_extraction", run_spot_extraction, take2),
            ("spot_zipping", run_spot_zipping, take2),
            ("tracing", run_chromatin_tracing, take2),
            ("spot_region_distances", run_frame_name_and_distance_application, take2), 
            (TRACING_QC_STAGE_NAME, qc_label_and_filter_traces, take2),
            ("spot_counts_visualisation__locus_specific", plot_spot_counts, take1_with_spot_type(SpotType.LOCUS_SPECIFIC)), 
            ("pairwise_distances__locus_specific", compute_locus_pairwise_distances, take1),
            ("pairwise_distances__regional", compute_region_pairwise_distances, take1),
        ]

    def stages(self) -> List[Callable]:
        return [
            pypiper.Stage(func=fxn, f_args=get_args_from_conf_and_imgs(self.config_file, self.images_folder), name=name) 
            for name, fxn, get_args_from_conf_and_imgs in self.name_fun_getargs_bundles()
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
