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
from looptrace.Tracer import Tracer, run_frame_name_and_distance_application
from looptrace.conversion_to_zarr import one_to_one as run_zarr_production

from pipeline_precheck import workflow as pretest
from nuc_label import workflow as run_nuclei_detection
from nuc_label_qc import workflow as run_nuclei_visualisation
from extract_exp_psf import workflow as run_psf_extraction
from run_bead_roi_generation import workflow as gen_all_bead_rois
from analyse_detected_bead_rois import workflow as run_all_bead_roi_detection_analysis
from decon import workflow as run_deconvolution
from drift_correct_accuracy_analysis import workflow as run_drift_correction_analysis, run_visualisation as run_drift_correction_accuracy_visualisation
from detect_spots import workflow as run_spot_detection
from assign_spots_to_nucs import workflow as run_spot_nucleus_filtration
from extract_spots_table import workflow as run_spot_bounding
from extract_spots import workflow as run_spot_extraction
from zip_spot_image_files_for_tracing import workflow as run_spot_zipping
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


def partition_bead_rois(rounds_config: ExtantFile, params_config: ExtantFile, images_folder: ExtantFolder):
    """Run the bead ROIs partitioning program / pipeline step."""
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder)
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


def run_spot_proximity_filtration(rounds_config: ExtantFile, params_config: ExtantFile, images_folder: ExtantFolder) -> None:
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder)
    min_spot_sep = H.minimum_spot_separation
    if min_spot_sep <= 0:
        print(f"No spot filtration on proximity to be done, as minimum separation is nonpositive: {min_spot_sep}")
        return
    
    # Command construction, printing, and execution
    prog_path = f"{LOOPTRACE_JAVA_PACKAGE}.LabelAndFilterRois"
    cmd_parts = [
        "java", 
        "-cp",
        str(LOOPTRACE_JAR_PATH),
        prog_path, 
        "--configuration",
        str(rounds_config.path),
        "--spotsFile",
        str(H.raw_spots_file),
        "--driftFile", 
        str(H.drift_correction_file__fine),
        "--unfilteredOutputFile",
        str(H.proximity_labeled_spots_file_path),
        "--filteredOutputFile",
        str(H.proximity_filtered_spots_file_path),
        "--handleExtantOutput",
        "OVERWRITE" # TODO: parameterise this, see: https://github.com/gerlichlab/looptrace/issues/142
    ]
    print(f"Running spot filtering on proximity: {' '.join(cmd_parts)}")
    subprocess.check_call(cmd_parts)


def plot_spot_counts(rounds_config: ExtantFile, params_config: ExtantFile, spot_type: "SpotType") -> None:
    """Plot the unfiltered and filtered counts of spots (regional or locus-specific).
    
    Plot heatmaps stratifying spot counts by timepoint and by field-of-view, and 
    plot grouped (by timepoint) barchart with unfiltered and filtered side-by-side 
    for each timepoint, with count summed across all fields of view.

    Arguments
    ---------
    rounds_config : gertils.ExtantFile
        Path to the looptrace imaging rounds configuration file
    params_config : gertils.ExtantFile
        Path to the looptrace parameters configuration file
    spot_type : SpotType
        Enumeration value indicating the type of spot counts data to plot
    
    Raises
    ------
    FileNotFoundError: if the path to the analysis script to run isn't found as a file
    ValueError: if the given spot_type value isn't recognised
    """
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config)
    output_folder = H.analysis_path
    analysis_script_file = Path(os.path.dirname(__file__)) / "spot_counts_visualisation.R"
    if not analysis_script_file.is_file():
        raise FileNotFoundError(f"Missing regional spot counts plot script: {analysis_script_file}")
    if spot_type == SpotType.REGIONAL:
        filtered = H.proximity_filtered_spots_file_path
        extra_files = [
            "--unfiltered-spots-file", str(H.raw_spots_file), 
            "--nuclei-filtered-spots-file", str(H.nuclei_filtered_spots_file_path),
            ]
        probe_name_extra = ["--probe-names"] + H.frame_names
    elif spot_type == SpotType.LOCUS_SPECIFIC:
        filtered = H.traces_file_qc_filtered
        extra_files = []
        probe_name_extra = []
    else:
        raise ValueError(f"Illegal spot_type for plotting spot counts: {spot_type}")
    cmd_parts = [
        "Rscript", 
        str(analysis_script_file), 
        "--filtered-spots-file",
        str(filtered),
        "--spot-file-type", 
        spot_type.value,
        "-o", 
        output_folder, 
        ] + extra_files + probe_name_extra
    print(f"Running spot count plotting command: {' '.join(cmd_parts)}")
    return subprocess.check_call(cmd_parts)


def qc_locus_spots_and_prep_points(rounds_config: ExtantFile, params_config: ExtantFile, images_folder: ExtantFolder) -> None:
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder)
    prog_path = f"{LOOPTRACE_JAVA_PACKAGE}.LabelAndFilterLocusSpots"
    cmd_parts = [
        "java", 
        "-cp",
        str(LOOPTRACE_JAR_PATH),
        prog_path,
        "--configuration", 
        str(rounds_config.path),
        prog_path, 
        "--tracesFile",
        str(H.traces_path_enriched),
        "--roiPixelsZ",
        str(H.roi_image_size.z),
        "--roiPixelsY",
        str(H.roi_image_size.y),
        "--roiPixelsX",
        str(H.roi_image_size.x),
        "--maxDistanceToRegionCenter", 
        str(H.config[MAX_DISTANCE_SPOT_FROM_REGION_NAME]),
        "--minSNR",
        str(H.config[SIGNAL_NOISE_RATIO_NAME]),
        "--maxSigmaXY",
        str(H.config[SIGMA_XY_MAX_NAME]),
        "--maxSigmaZ",
        str(H.config[SIGMA_Z_MAX_NAME]),
        "--pointsDataOutputFolder", 
        str(H.locus_spot_images_root_path),
    ]    
    print(f"Running QC filtering of tracing supports: {' '.join(cmd_parts)}")
    subprocess.check_call(cmd_parts)


def compute_locus_pairwise_distances(rounds_config: ExtantFile, params_config: ExtantFile) -> None:
    """Run the locus pairwise distances computation program."""
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config)
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


def compute_region_pairwise_distances(rounds_config: ExtantFile, params_config: ExtantFile) -> None:
    """Run the regional pairwise distances computation program."""
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config)
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


def drift_correct_nuclei(rounds_config: ExtantFile, params_config: ExtantFile, images_folder: ExtantFolder) -> Path:
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder)
    N = NucDetector(H)
    return N.coarse_drift_correction_workflow()


def prep_locus_specific_spots_visualisation(rounds_config: ExtantFile, params_config: ExtantFile, images_folder: ExtantFolder) -> Tuple[Path, List[Path]]:
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder)
    T = Tracer(H)
    all_one_zarr = T.write_spot_images_subset_to_single_highly_nested_zarr()
    per_fov_zarr = T.write_all_spot_images_to_one_per_fov_zarr()
    return all_one_zarr, per_fov_zarr


class LooptracePipeline(pypiper.Pipeline):
    """Main looptrace processing pipeline"""

    def __init__(self, rounds_config: ExtantFile, params_config: ExtantFile, images_folder: ExtantFolder, output_folder: ExtantFolder, **pl_mgr_kwargs: Any) -> None:
        self.rounds_config = rounds_config
        self.params_config = params_config
        self.images_folder = images_folder
        super(LooptracePipeline, self).__init__(name=PIPE_NAME, outfolder=str(output_folder.path), **pl_mgr_kwargs)

    @staticmethod
    def name_fun_getargs_bundles() -> List[Tuple[str, callable, Callable[[Tuple[ExtantFile, ExtantFolder]], Union[Tuple[ExtantFile], Tuple[ExtantFile, ExtantFolder]]]]]:
        take2 = lambda rounds_config, params_config, _: (rounds_config, params_config, )
        take2_with_spot_type = lambda spot_type: (lambda rounds_config, params_config, _3: take2(rounds_config, params_config, _3) + (spot_type, ))
        take3 = lambda rounds_config, params_config, images_folder: (rounds_config, params_config, images_folder)
        return [
            ("pipeline_precheck", pretest, take2),
            ("zarr_production", run_zarr_production, take3),
            ("nuclei_detection", run_nuclei_detection, take3),
            ("nuclei_visualisation", run_nuclei_visualisation, take3), 
            ("nuclei_drift_correction", drift_correct_nuclei, take3),
            ("psf_extraction", run_psf_extraction, take3),
            (DECON_STAGE_NAME, run_deconvolution, take3), # Really just for denoising, no need for structural disambiguation
            ("drift_correction__coarse", run_coarse_drift_correction, take3), 
            ("bead_roi_generation", gen_all_bead_rois, take3), # Find/define all the bead ROIs in each (FOV, frame) pair.
            # Count detected bead ROIs for each timepoint, mainly to see if anything went awry during some phase of the imaging, e.g. air bubble.
            ("bead_roi_detection_analysis", run_all_bead_roi_detection_analysis, take3),
            ("bead_roi_partition", partition_bead_rois, take3),
            ("drift_correction__fine", run_fine_drift_correction, take3),
            ("drift_correction_accuracy_analysis", run_drift_correction_analysis, take3), 
            ("drift_correction_accuracy_visualisation", run_drift_correction_accuracy_visualisation, lambda _1, params_config, _3: (params_config, )), 
            (SPOT_DETECTION_STAGE_NAME, run_spot_detection, take3), # generates *_rois.csv (regional spots)
            ("spot_proximity_filtration", run_spot_proximity_filtration, take3),
            ("spot_nucleus_filtration", run_spot_nucleus_filtration, take3), 
            ("spot_counts_visualisation__regional", plot_spot_counts, take2_with_spot_type(SpotType.REGIONAL)), 
            ("spot_bounding", run_spot_bounding, take3), # computes pad_x_min, etc.; writes *_dc_rois.csv (much bigger, since regional spots x frames)
            ("spot_extraction", run_spot_extraction, take3),
            ("spot_zipping", run_spot_zipping, lambda _1, params_config, images_folder: (params_config, images_folder)),
            ("tracing", run_chromatin_tracing, take3),
            ("spot_region_distances", run_frame_name_and_distance_application, take3), 
            (TRACING_QC_STAGE_NAME, qc_locus_spots_and_prep_points, take3),
            ("spot_counts_visualisation__locus_specific", plot_spot_counts, take2_with_spot_type(SpotType.LOCUS_SPECIFIC)), 
            ("pairwise_distances__locus_specific", compute_locus_pairwise_distances, take2),
            ("pairwise_distances__regional", compute_region_pairwise_distances, take2),
            ("locus_specific_spots_visualisation_data_prep", prep_locus_specific_spots_visualisation, take3),
        ]

    def stages(self) -> List[Callable]:
        return [
            pypiper.Stage(func=fxn, f_args=get_args_from_conf_and_imgs(self.rounds_config, self.params_config, self.images_folder), name=name) 
            for name, fxn, get_args_from_conf_and_imgs in self.name_fun_getargs_bundles()
        ]


def parse_cli(args: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A pipeline to process microscopy imaging data to trace chromatin fiber with FISH probes")
    parser.add_argument("--rounds-config", type=ExtantFile.from_string, required=True, help="Path to the imaging rounds configuration file")
    parser.add_argument("--params-config", type=ExtantFile.from_string, required=True, help="Path to the parameters configuration file")
    parser.add_argument("-I", "--images-folder", type=ExtantFolder.from_string, required=True, help="Path to the root folder with imaging data to process")
    parser.add_argument("--pypiper-folder", type=ExtantFolder.from_string, required=True, help="Path to folder for pypiper output")
    parser.add_argument(NO_TEE_LOGS_OPTNAME, action="store_true", help="Do not tee logging output from pypiper manager")
    parser = pypiper.add_pypiper_args(
        parser, 
        groups=("pypiper", "checkpoint"),
        args=("start-point", ),
        )
    return parser.parse_args(args)


def init(opts: argparse.Namespace) -> LooptracePipeline:
    kwargs = {
        "rounds_config": opts.rounds_config, 
        "params_config": opts.params_config, 
        "images_folder": opts.images_folder, 
        "output_folder": opts.pypiper_folder,
        }
    if opts.do_not_tee_logs:
        kwargs["multi"] = True
    logger.info(f"Building {PIPE_NAME} pipeline, using images from {opts.images_folder.path}")
    return LooptracePipeline(**kwargs)


def main(cmdl):
    opts = parse_cli(cmdl)
    pipeline = init(opts)
    logger.info("Running pipeline")
    pipeline.run(start_point=opts.start_point, stop_after=opts.stop_after)
    pipeline.wrapup()


if __name__ == "__main__":
    main(sys.argv[1:])
