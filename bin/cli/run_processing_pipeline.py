"""Simple end-to-end processing pipeline for chromatin tracing data"""

import argparse
from enum import Enum
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import *

import dask.array as da
from expression import Option, result
from expression.collections import Seq
from gertils import ExtantFile, ExtantFolder
import pandas as pd
import pypiper
import yaml

from looptrace import *
from looptrace.Drifter import coarse_correction_workflow, fine_correction_workflow as run_fine_drift_correction
from looptrace.ImageHandler import BeadRoisFilenameSpecification, ImageHandler
from looptrace.NucDetector import NucDetector
from looptrace.Tracer import Tracer, run_timepoint_name_and_distance_application
from looptrace.bead_roi_generation import FAIL_CODE_COLUMN_NAME
from looptrace.configuration import KEY_FOR_SEPARATION_NEEDED_TO_NOT_MERGE_ROIS
from looptrace.conversion_to_zarr import one_to_one as run_zarr_production
from looptrace.image_processing_functions import extract_labeled_centroids
from looptrace.integer_naming import parse_field_of_view_one_based_from_position_name_representation
from looptrace.trace_metadata import PotentialTraceMetadata

from pipeline_precheck import workflow as pretest
from nuc_label import workflow as run_nuclei_detection
from extract_exp_psf import workflow as run_psf_extraction
from run_bead_roi_generation import workflow as gen_all_bead_rois
from analyse_detected_bead_rois import workflow as run_all_bead_roi_detection_analysis
from decon import workflow as run_deconvolution
from drift_correct_accuracy_analysis import workflow as run_drift_correction_analysis, run_visualisation as run_drift_correction_accuracy_visualisation
from detect_spots import workflow as run_spot_detection
from assign_spots_to_nucs import (
    NUC_LABEL_COL, 
    add_nucleus_labels, 
    workflow as run_spot_nucleus_assignment,
)
from partition_regional_spots_for_locus_spots_visualisation import workflow as prep_regional_spots_visualisation
from extract_spots_table import workflow as run_spot_bounding
from extract_spots import workflow as run_spot_extraction
from zip_spot_image_files_for_tracing import workflow as run_spot_zipping
from tracing import workflow as run_chromatin_tracing
from locus_spot_visualisation_data_preparation import workflow as get_locus_spot_data_file_src_dst_pairs
from run_signal_analysis import workflow as signal_analysis_workflow


DECON_STAGE_NAME = "deconvolution"
NO_TEE_LOGS_OPTNAME = "--do-not-tee-logs"
PIPE_NAME = "looptrace"
SPOT_DETECTION_STAGE_NAME = "spot_detection"
TRACING_QC_STAGE_NAME = "tracing_QC"

FieldOfViewName: TypeAlias = str


class SpotType(Enum):
    REGIONAL = "regional"
    LOCUS_SPECIFIC = "locus-specific"


def partition_bead_rois(rounds_config: ExtantFile, params_config: ExtantFile, images_folder: ExtantFolder):
    """Run the bead ROIs partitioning program / pipeline step."""
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder)
    prog_path = f"{LOOPTRACE_JAVA_PACKAGE}.PartitionIndexedDriftCorrectionBeadRois"
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
    logging.info(f"Running bead ROI partitioning: {' '.join(cmd_parts)}")
    subprocess.check_call(cmd_parts)


def run_spot_proximity_filtration(rounds_config: ExtantFile, params_config: ExtantFile, images_folder: ExtantFolder) -> None:
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder)

    # Command construction, printing, and execution
    prog_path = f"{LOOPTRACE_JAVA_PACKAGE}.FilterRoisByProximity"
    cmd_parts = [
        "java", 
        "-cp",
        str(LOOPTRACE_JAR_PATH),
        prog_path, 
        "--configuration",
        str(rounds_config.path),
        "--spotsFile",
        str(H.spot_merge_results_file),
        "--driftFile", 
        str(H.drift_correction_file__fine),
        "--fileForDiscards",
        str(H.proximity_rejected_spots_file_path),
        "--fileForKeepers",
        str(H.proximity_accepted_spots_file_path),
        "--overwrite",
    ]
    logging.info(f"Running spot filtering on proximity: {' '.join(cmd_parts)}")
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
        filtered = H.proximity_accepted_spots_file_path
        extra_files = [
            "--unfiltered-spots-file", str(H.raw_spots_file), 
            "--nuclei-filtered-spots-file", str(H.nuclei_filtered_spots_file_path),
            ]
        probe_name_extra = ["--probe-names"] + H.timepoint_names
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
    logging.info(f"Running spot count plotting command: {' '.join(cmd_parts)}")
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
        str(H.locus_spots_visualisation_folder),
        "--overwrite",
    ]    
    logging.info(f"Running QC filtering of tracing supports: {' '.join(cmd_parts)}")
    subprocess.check_call(cmd_parts)


def annotate_traces(rounds_config: ExtantFile, params_config: ExtantFile, images_folder: ExtantFolder) -> Path:
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder)
    T = Tracer(H)
    return T.write_traces_file()


def assign_trace_ids(rounds_config: ExtantFile, params_config: ExtantFile) -> Path:
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config)
    cmd_parts = [
        "java", 
        "-cp",
        str(LOOPTRACE_JAR_PATH),
        f"{LOOPTRACE_JAVA_PACKAGE}.AssignTraceIds",
        "--configuration",
        str(rounds_config.path),
        "--pixels",
        build_pixels_config(H),
        "--roisFile",
        str(H.nuclei_labeled_spots_file_path if H.spot_in_nuc else H.proximity_accepted_spots_file_path),
        "--outputFile", 
        str(H.rois_with_trace_ids_file),
        "--skipsFile", 
        str(H.trace_id_assignment_skipped_rois_file),
    ]
    logging.info(f"Running distance computation command: {' '.join(cmd_parts)}")
    return subprocess.check_call(cmd_parts)


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
    logging.info(f"Running distance computation command: {' '.join(cmd_parts)}")
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
        str(H.spots_for_voxels_definition_file),
        "--driftFile", 
        str(H.drift_correction_file__fine),
        "--pixels",
        build_pixels_config(H),
        "-O", 
        H.analysis_path,
    ]
    logging.info(f"Running distance computation command: {' '.join(cmd_parts)}")
    return subprocess.check_call(cmd_parts)


def build_pixels_config(H: ImageHandler) -> str:
    return f"{{ x: {H.nanometers_per_pixel_xy} nm, y: {H.nanometers_per_pixel_xy} nm, z: {H.nanometers_per_pixel_z} nm }}"


def drift_correct_nuclei(rounds_config: ExtantFile, params_config: ExtantFile, images_folder: ExtantFolder) -> Path:
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder)
    N = NucDetector(H)
    return N.coarse_drift_correction_workflow()


def get_merge_rules_for_tracing(H: ImageHandler) -> Mapping[str, object]:
    return H.config["mergeRulesForTracing"]


def prep_locus_specific_spots_visualisation(rounds_config: ExtantFile, params_config: ExtantFile, images_folder: ExtantFolder) -> list[Path]:
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder)
    T = Tracer(H)
    
    try:
        raw_config_metadata: Mapping[str, object] = get_merge_rules_for_tracing(H)
    except KeyError:
        # This was the original case, before the addition of merger of ROIs to tracing structures.
        # NB: This does no accounting of structural difference which occurs when there's merger of ROIs for tracing.
        #     That is, each trace in this case will correspond to exactly one regional timepoint, but 
        #     each regional timepoint may have a different number of locus-specific timepoints, and therefore 
        #     when traces are stacked together for a field of view, independent of the regional timepoints, 
        #     there will quite likely be traces with different numbers of locus-specific timepoints supporting them. 
        #     The function called here takes care of "smoothing out" any otherwise ragged arrays, by simply filling 
        #     them from the back with all-0s arrays. Specifically, what's added is an array of all-0s vovels, 
        #     with the length of this array equal to the difference between the maximum number of timepoints possible 
        #     for a particular trace (the one whose image stack is having voxels appended), and the maximum number of 
        #     timepoints possible for ANY trace, regardless of which regional timepoint it's associated with.
        logging.info("No ROI merger for tracing to account for during preparation of locus spot visualisation")
        return T.write_all_spot_images_to_viewable_stacks(metadata=None)
    else:
        logging.info("Using ROI merge rules for tracing to prep locus spots visualisation")
        match Option.of_optional(raw_config_metadata.get("groups"))\
            .to_result(["Missing groups subsection in tracing merge rules section of imaging rounds config"])\
            .bind(PotentialTraceMetadata.from_mapping):
            case result.Result(tag="error", error=problem_messages):
                logging.error("Failed to parse potential trace metadata!")
                for msg in problem_messages:
                    logging.error(msg)
                raise RuntimeError(f"{len(problem_messages)} error(s) parsing potential trace metadata: {problem_messages}")
            case result.Result(tag="ok", ok=metadata):
                return T.write_all_spot_images_to_viewable_stacks(metadata=metadata)


def prep_nuclear_masks_data(rounds_config: ExtantFile, params_config: ExtantFile, images_folder: ExtantFolder) -> Dict[str, Path]:
    """Write simple CSV data for visualisation of nuclear masks with napari plugin."""
    def raise_error(msg: str):
        raise Exception(msg)
    
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder)
    N = NucDetector(H)
    result: Dict[int, Path] = {}
    for fov, img in N.iterate_over_pairs_of_fov_and_mask_image():
        logging.info(f"Computing nuclear mask centroids for FOV: {fov}")
        table = extract_labeled_centroids(img)
        logging.info(f"Finished nuclear mask centroids for FOV: {fov}")
        cleaned_pos_name: str = fov.removesuffix(".zarr")
        # Note that in what follows, we ignore the actual value from the parse; here we just care about proof that the 
        # cleaned value corresponds to one with a valid parse as a 1-based FOV.
        fp: Path = \
            parse_field_of_view_one_based_from_position_name_representation(cleaned_pos_name)\
                .map_error(lambda msg: f"Failed to parse semantic and value from text ({cleaned_pos_name}): {msg}")\
                .map(lambda _: H.nuclear_masks_visualisation_data_path / f"{cleaned_pos_name}.nuclear_masks.csv")\
                .default_with(raise_error)
        logging.info(f"Writing data file for nuclei visualisation to file: {fp}")
        fp.parent.mkdir(exist_ok=True)
        table.to_csv(fp)
        result[fov] = fp
    return result


def move_nuclear_masks_visualisation_data(rounds_config: ExtantFile, params_config: ExtantFile, images_folder: ExtantFolder) -> ExtantFolder:
    """Set up the nuclei centers data, relative to the nuclei image and masks data, such that they can be viewed with the Napari plugin."""
    # To avoid images read, don't pass images folder, since it's unneeded for this actual handler.
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config)
    src = H.nuclear_masks_visualisation_data_path
    dst = images_folder.path / ("_" + src.name) # Here's where we actually use the images_folder for this step.
    logging.info("Copying nuclear mask visualisation data: %s --> %s", src, dst)
    try:
        shutil.copytree(src, dst)
    except Exception as e:
        logging.warning(f"Error during copy of {src} to {dst}")
        old_size_by_name = {f.name: os.path.getsize(f) for f in src.iterdir()}
        new_size_by_name = {f.name: os.path.getsize(f) for f in dst.iterdir()}
        if old_size_by_name == new_size_by_name:
            logging.warning(f"Contents of {dst} appear to match those of {src} -- all good!")
        else:
            raise
    return ExtantFolder(dst)


def run_regional_spot_viewing_prep(rounds_config: ExtantFile, params_config: ExtantFile) -> dict[str, list[Path]]:
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config)
    prep_regional_spots_visualisation(
        output_folder=H.regional_spots_visualisation_data_path,
        spots_files=[
            H.spot_merge_contributors_file, 
            H.proximity_rejected_spots_file_path, 
            H.rois_with_trace_ids_file,
        ],
    )


def run_locus_spot_viewing_prep(rounds_config: ExtantFile, params_config: ExtantFile) -> None:
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config)
    src_dst_pairs: list[tuple[Path, Path]] = get_locus_spot_data_file_src_dst_pairs(infolder=H.locus_spots_visualisation_folder)
    for src, dst in src_dst_pairs:
        logging.debug("Moving: %s --> %s", src, dst)
        shutil.move(src, dst)


def validate_imaging_rounds_config(rounds_config: ExtantFile) -> int:
    cmd_parts = [
        "java", 
        "-cp",
        str(LOOPTRACE_JAR_PATH),
        f"{LOOPTRACE_JAVA_PACKAGE}.ValidateImagingRoundsConfig",
        str(rounds_config.path),
    ]
    logging.info(f"Running imaging rounds validation: {' '.join(cmd_parts)}")
    return subprocess.check_call(cmd_parts)


def run_coarse_drift_correction(
    rounds_config: ExtantFile, 
    params_config: ExtantFile, 
    images_folder: ExtantFolder,
) -> Path:
    logging.info("Checking for coarse drift correction parameters in config file: %s", params_config.path)
    with params_config.path.open(mode="r") as fh:
        conf = yaml.safe_load(fh)
    extra_kwargs: dict[str, object] = {}
    for config_key, param_name in [("coarse_dc_backend", "joblib_backend"), ("coarse_dc_cpu_count", "n_jobs")]:
        config_val = conf.get(config_key)
        if config_val is not None:
            logging.debug("%s found; will set %s to %s", config_key, param_name, str(config_val))
            extra_kwargs[param_name] = config_val
        else:
            logging.debug("Not in config, ignoring: %s", config_key)
    return coarse_correction_workflow(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder, **extra_kwargs)


def run_spot_merge_determination(rounds_config: ExtantFile, params_config: ExtantFile) -> None:
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config)

    # Command construction, printing, and execution
    prog_path = f"{LOOPTRACE_JAVA_PACKAGE}.DetermineRoiMerge"
    cmd_parts = [
        "java", 
        "-cp",
        str(LOOPTRACE_JAR_PATH),
        prog_path, 
        "-I",
        str(H.raw_spots_file),
        "-D", 
        str(H.config[KEY_FOR_SEPARATION_NEEDED_TO_NOT_MERGE_ROIS]),
        "-O",
        str(H.pre_merge_spots_file),
        "--overwrite",
    ]
    logging.info(f"Running spot merge determination: {' '.join(cmd_parts)}")
    subprocess.check_call(cmd_parts)


def run_spot_merge_execution(rounds_config: ExtantFile, params_config: ExtantFile) -> None:
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config)

    # Command construction, printing, and execution
    prog_path = f"{LOOPTRACE_JAVA_PACKAGE}.MergeRois"
    cmd_parts = [
        "java", 
        "-cp",
        str(LOOPTRACE_JAR_PATH),
        prog_path, 
        "-I",
        str(H.pre_merge_spots_file),
        "--mergeContributorsFile",
        str(H.spot_merge_contributors_file),
        "--mergeResultsFile",
        str(H.spot_merge_results_file),
        "--overwrite",
    ]
    logging.info(f"Running spot merge execution: {' '.join(cmd_parts)}")
    subprocess.check_call(cmd_parts)


def filter_spots_for_nuclei(rounds_config: ExtantFile, params_config: ExtantFile) -> None:
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config)
    if not H.spot_in_nuc:
        logging.info("spot_in_nuc is False, nothing to do, skipping nuclei-based filtration")
        return
    rois_file: Path = H.rois_with_trace_ids_file
    logging.info(f"Reading ROIs file for filtration: {rois_file}")
    rois: pd.DataFrame = pd.read_csv(rois_file, index_col=False)
    logging.debug(f"Initial ROI count: {rois.shape[0]}")
    rois = rois[rois[NUC_LABEL_COL] != 0]
    logging.debug(f"ROIs remaining after filtration through nuclei: {rois.shape[0]}")
    logging.debug(f"Writing ROIs: {H.nuclei_filtered_spots_file_path}")
    rois.to_csv(H.nuclei_filtered_spots_file_path, index=False)
    logging.info("Done with nuclei-based spot filtration")


def filter_beads(
    rounds_config: ExtantFile, 
    params_config: ExtantFile, 
    images_folder: ExtantFolder,
) -> None:
    def get_spec_opt(fn: str) -> Option[tuple[BeadRoisFilenameSpecification, Path]]:
        return Option.of_optional(BeadRoisFilenameSpecification.from_filename(fn)).map(lambda spec: (spec, H.bead_rois_path / fn))
    
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder)
    fov_mask_pairs: list[tuple[FieldOfViewName, da.Array]] = list(zip(H.image_lists[H.spot_input_name], H.images[NucDetector.MASKS_KEY], strict=True))
    
    spec_file_pairs: list[tuple[BeadRoisFilenameSpecification, Path]] = \
        Seq.of_iterable(os.listdir(H.bead_rois_path)).choose(get_spec_opt).to_list()
    
    logging.info("Bead files count: %d", len(spec_file_pairs))
    for bead_spec, rois_file in spec_file_pairs:
        fov_idx: int = bead_spec.fov
        bead_rois: pd.DataFrame = pd.read_csv(rois_file, index_col=None)
        old_num_rois: int = bead_rois.shape[0]
        logging.info("Detected bead count: %d", old_num_rois)
        # After filtering by the fail code, we no longer need that column name.
        bead_rois = bead_rois[(bead_rois[FAIL_CODE_COLUMN_NAME] == "") | (bead_rois[FAIL_CODE_COLUMN_NAME].isna())].drop(columns=[FAIL_CODE_COLUMN_NAME])
        logging.info("QC-pass bead count: %d", bead_rois.shape[0])
        bead_rois: pd.DataFrame = add_nucleus_labels(
            rois_table=bead_rois, 
            mask_images=[fov_mask_pairs[fov_idx]],
            nuclei_drift_file=H.nuclei_coarse_drift_correction_file,
            spots_drift_file=H.drift_correction_file__coarse,
            timepoint=Option.Some(bead_spec.timepoint),
            remove_zarr_suffix=False,
        )
        
        # For beads (not FISH spots), we want things OUTside the nucelus; after filtration, we can drop the nucleus label.
        bead_rois = bead_rois[bead_rois[NUC_LABEL_COL] == 0].drop(columns=[NUC_LABEL_COL])
        new_num_rois: int = bead_rois.shape[0]
        filtered_rois_file: Path = rois_file.with_suffix(".filtered.csv")
        logging.info("%s -> %s: %d --> %d", rois_file, filtered_rois_file, old_num_rois, new_num_rois)
        bead_rois.to_csv(filtered_rois_file, index=False)


def discard_spots_close_to_beads(rounds_config: ExtantFile, params_config: ExtantFile) -> None:
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config)
    
    # Command construction, printing, and execution
    prog_path = f"{LOOPTRACE_JAVA_PACKAGE}.FilterSpotsByBeads"
    cmd_parts = [
        "java", 
        "-cp",
        str(LOOPTRACE_JAR_PATH),
        prog_path, 
        "--spotsFolder",
        str(H.fish_spots_folder),
        "--beadsFolder",
        str(H.bead_rois_path),
        "--driftFile",
        str(H.drift_correction_file__fine),
        "--filteredOutputFile",
        str(H.raw_spots_file),
        "--distanceThreshold",
        str(H.config["beadSpotProximityDistanceInNanometers"]),
        "--spotlessTimepoint",
        str(H.bead_timepoint_for_spot_filtration.get),
        "--pixels",
        build_pixels_config(H),
        "--overwrite"
    ]
    logging.info(f"Filtering FISH spots by proximity to beads: {' '.join(cmd_parts)}")
    subprocess.check_call(cmd_parts)


def validate_roi_mergers(rounds_config: ExtantFile, params_config: ExtantFile) -> None:
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config)

    same_timepoint_threshold = H.config[KEY_FOR_SEPARATION_NEEDED_TO_NOT_MERGE_ROIS]

    prog_path = f"{LOOPTRACE_JAVA_PACKAGE}.ValidateMergeDetermination"
    cmd_base_parts: list[str] = [
        "java", 
        "-cp", 
        str(LOOPTRACE_JAR_PATH), 
        prog_path,
        "--sameTimepointDistanceThreshold", 
        str(same_timepoint_threshold),
    ]

    if same_timepoint_threshold == 0:
        logging.info("ROI separation threshold for same-timepoint merge is 0, so no validation to do")
    else:
        logging.info(f"Same-timepoint mergers will be validated according to threshold: {same_timepoint_threshold}")
        cmd_parts = cmd_base_parts + [
            "--inputFile", 
            str(H.proximity_rejected_spots_file_path),
            "--mergeType", 
            "same",
        ]
        logging.info(f"Validating same-timepoint ROI mergers: {' '.join(cmd_parts)}")
        subprocess.check_call(cmd_parts)


class LooptracePipeline(pypiper.Pipeline):
    """Main looptrace processing pipeline"""

    def __init__(
            self, 
            rounds_config: ExtantFile, 
            params_config: ExtantFile, 
            images_folder: ExtantFolder, 
            output_folder: ExtantFolder, 
            signal_config: Option[ExtantFile],
            **pl_mgr_kwargs: Any,
        ) -> None:
        self.rounds_config = rounds_config
        self.params_config = params_config
        self.images_folder = images_folder
        self.signal_config = signal_config
        super(LooptracePipeline, self).__init__(name=PIPE_NAME, outfolder=str(output_folder.path), **pl_mgr_kwargs)

    def stages(self) -> list[pypiper.Stage]:
        rounds_params = {"rounds_config": self.rounds_config, "params_config": self.params_config}
        rounds_params_images = {**rounds_params, "images_folder": self.images_folder}
        return [
            pypiper.Stage(name="imaging_rounds_validation", func=validate_imaging_rounds_config, f_kwargs={"rounds_config": self.rounds_config}),
            pypiper.Stage(name="pipeline_precheck", func=pretest, f_kwargs=rounds_params),
            pypiper.Stage(name="zarr_production", func=run_zarr_production, f_kwargs=rounds_params_images),
            pypiper.Stage(name="psf_extraction", func=run_psf_extraction, f_kwargs=rounds_params_images),
            # Really just for denoising, no need for structural disambiguation
            pypiper.Stage(name=DECON_STAGE_NAME, func=run_deconvolution, f_kwargs=rounds_params_images),
            pypiper.Stage(name="nuclei_detection", func=run_nuclei_detection, f_kwargs=rounds_params_images),
            pypiper.Stage(name="nuclei_drift_correction", func=drift_correct_nuclei, f_kwargs=rounds_params_images),
            pypiper.Stage(name="nuclear_masks_visualisation_data_prep", func=prep_nuclear_masks_data, f_kwargs=rounds_params_images),
            pypiper.Stage(
                name="move_nuclear_masks_visualisation_data", 
                func=move_nuclear_masks_visualisation_data, 
                f_kwargs=rounds_params_images,
                nofail=True,
            ),
            pypiper.Stage(name="drift_correction__coarse", func=run_coarse_drift_correction, f_kwargs=rounds_params_images), 
            # Find/define all the bead ROIs in each (FOV, timepoint) pair.
            pypiper.Stage(name="bead_roi_generation", func=gen_all_bead_rois, f_kwargs=rounds_params_images),
            # Count detected bead ROIs for each timepoint, mainly to see if anything went awry during some phase of the imaging, e.g. air bubble.
            pypiper.Stage(name="bead_roi_detection_analysis", func=run_all_bead_roi_detection_analysis, f_kwargs=rounds_params_images),
            pypiper.Stage(name="bead_filtration", func=filter_beads, f_kwargs=rounds_params_images),
            pypiper.Stage(name="bead_roi_partition", func=partition_bead_rois, f_kwargs=rounds_params_images),
            pypiper.Stage(name="drift_correction__fine", func=run_fine_drift_correction, f_kwargs=rounds_params_images),
            pypiper.Stage(name="drift_correction_accuracy_analysis", func=run_drift_correction_analysis, f_kwargs=rounds_params_images), 
            pypiper.Stage(
                name="drift_correction_accuracy_visualisation", 
                func=run_drift_correction_accuracy_visualisation, 
                f_kwargs={"params_config": self.params_config},
            ), 
            pypiper.Stage(name=SPOT_DETECTION_STAGE_NAME, func=run_spot_detection, f_kwargs=rounds_params_images), # generates *_rois.csv (regional spots)
            pypiper.Stage(
                name="spot_bead_proximity_filtration", 
                func=discard_spots_close_to_beads, 
                f_kwargs=rounds_params, # Images are not needed since only bead coordinates and spot coordinates are needed.
            ), 
            pypiper.Stage(name="spot_merge_determination", func=run_spot_merge_determination, f_kwargs=rounds_params),
            pypiper.Stage(name="spot_merge_execution", func=run_spot_merge_execution, f_kwargs=rounds_params),
            pypiper.Stage(name="spot_proximity_filtration", func=run_spot_proximity_filtration, f_kwargs=rounds_params_images),
            pypiper.Stage(
                name="spot_nucleus_assignment", 
                func=run_spot_nucleus_assignment, 
                f_kwargs={"remove_zarr_suffix": True, **rounds_params_images}, # Images are needed since H.image_lists is iterated in workflow.
            ), 
            pypiper.Stage(
                name="trace_id_assignment",
                func=assign_trace_ids,
                f_kwargs=rounds_params,
            ),
            pypiper.Stage(
                name="spot_nuclei_filtration", 
                func=filter_spots_for_nuclei,
                f_kwargs=rounds_params, # Images are NOT needed here since the actual labeling/assignment has already been done.
            ),
            pypiper.Stage(
                name="regional_spots_visualisation_data_prep", 
                func=run_regional_spot_viewing_prep, 
                f_kwargs=rounds_params,
                nofail=True,
            ),
            pypiper.Stage(
                name="spot_counts_visualisation__regional", 
                func=plot_spot_counts, 
                f_kwargs={**rounds_params, "spot_type": SpotType.REGIONAL},
                nofail=True,
            ), 
            # computes pad_x_min, etc.; writes *_dc_rois.csv (much bigger, since regional spots x timepoints)
            pypiper.Stage(name="spot_bounding", func=run_spot_bounding, f_kwargs=rounds_params_images),
            pypiper.Stage(name="spot_extraction", func=run_spot_extraction, f_kwargs=rounds_params_images),
            pypiper.Stage(
                name="spot_zipping", 
                func=run_spot_zipping, 
                f_kwargs={"params_config": self.params_config, "images_folder": self.images_folder, "is_background": False},
            ),
            pypiper.Stage(
                name="spot_background_zipping", 
                func=run_spot_zipping, 
                f_kwargs={"params_config": self.params_config, "images_folder": self.images_folder, "is_background": True},
            ),
            pypiper.Stage(name="tracing", func=run_chromatin_tracing, f_kwargs=rounds_params_images), # Compute the 3D fits.
            pypiper.Stage(name="trace_annotation", func=annotate_traces, f_kwargs=rounds_params_images), # Put the 3D with each detected spot.
            pypiper.Stage(name="spot_region_distances", func=run_timepoint_name_and_distance_application, f_kwargs=rounds_params_images), 
            pypiper.Stage(name=TRACING_QC_STAGE_NAME, func=qc_locus_spots_and_prep_points, f_kwargs=rounds_params_images),
            pypiper.Stage(
                name="spot_counts_visualisation__locus_specific", 
                func=plot_spot_counts, 
                f_kwargs={**rounds_params, "spot_type": SpotType.LOCUS_SPECIFIC},
                nofail=True,
            ), 
            pypiper.Stage(
                name="pairwise_distances__locus_specific", 
                func=compute_locus_pairwise_distances, 
                f_kwargs=rounds_params,
            ),
            pypiper.Stage(
                name="pairwise_distances__regional", 
                func=compute_region_pairwise_distances, 
                f_kwargs=rounds_params,
                nofail=True,
            ),
            pypiper.Stage(
                name="locus_specific_spots_visualisation_data_prep", 
                func=prep_locus_specific_spots_visualisation, 
                f_kwargs=rounds_params_images,
            ),
            pypiper.Stage(
                name="cross_channel_signal_analysis",
                func=signal_analysis_workflow,
                f_kwargs={**rounds_params_images, "maybe_signal_config": self.signal_config},
            ),
            pypiper.Stage(
                name="merge_validation", 
                func=validate_roi_mergers, 
                f_kwargs=rounds_params,
                nofail=True,
            )
        ]


def parse_cli(args: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A pipeline to process microscopy imaging data to trace chromatin fiber with FISH probes")
    parser.add_argument("--rounds-config", type=ExtantFile.from_string, required=True, help="Path to the imaging rounds configuration file")
    parser.add_argument("--params-config", type=ExtantFile.from_string, required=True, help="Path to the parameters configuration file")
    parser.add_argument("-I", "--images-folder", type=ExtantFolder.from_string, required=True, help="Path to the root folder with imaging data to process")
    parser.add_argument("--pypiper-folder", type=ExtantFolder.from_string, required=True, help="Path to folder for pypiper output")
    parser.add_argument("--signal-config", type=ExtantFile.from_string, help="Path to signal analysis config file, if using that feature")
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
        "signal_config": Option.of_obj(opts.signal_config),
        "images_folder": opts.images_folder, 
        "output_folder": opts.pypiper_folder,
        }
    if opts.do_not_tee_logs:
        kwargs["multi"] = True
    logging.info(f"Building {PIPE_NAME} pipeline, using images from {opts.images_folder.path}")
    return LooptracePipeline(**kwargs)


def main(cmdl):
    opts = parse_cli(cmdl)
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Building pipeline")
    pipeline = init(opts)
    logging.info("Running pipeline")
    pipeline.run(start_point=opts.start_point, stop_after=opts.stop_after)
    pipeline.wrapup()


if __name__ == "__main__":
    main(sys.argv[1:])
