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

from gertils import ExtantFile, ExtantFolder
import pandas as pd
import pypiper

from looptrace import *
from looptrace.Drifter import coarse_correction_workflow as run_coarse_drift_correction, fine_correction_workflow as run_fine_drift_correction
from looptrace.ImageHandler import ImageHandler
from looptrace.NucDetector import NucDetector
from looptrace.Tracer import Tracer, run_frame_name_and_distance_application
from looptrace.conversion_to_zarr import one_to_one as run_zarr_production
from looptrace.image_processing_functions import extract_labeled_centroids
from looptrace.integer_naming import get_position_name_short

from pipeline_precheck import workflow as pretest
from nuc_label import workflow as run_nuclei_detection
from extract_exp_psf import workflow as run_psf_extraction
from run_bead_roi_generation import workflow as gen_all_bead_rois
from analyse_detected_bead_rois import workflow as run_all_bead_roi_detection_analysis
from decon import workflow as run_deconvolution
from drift_correct_accuracy_analysis import workflow as run_drift_correction_analysis, run_visualisation as run_drift_correction_accuracy_visualisation
from detect_spots import workflow as run_spot_detection
from assign_spots_to_nucs import workflow as run_spot_nucleus_filtration
from partition_regional_spots_by_field_of_view import workflow as prep_regional_spots_visualisation
from extract_spots_table import workflow as run_spot_bounding
from extract_spots import workflow as run_spot_extraction
from zip_spot_image_files_for_tracing import workflow as run_spot_zipping
from tracing import workflow as run_chromatin_tracing


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
    logging.info(f"Running bead ROI partitioning: {' '.join(cmd_parts)}")
    subprocess.check_call(cmd_parts)


def run_spot_proximity_filtration(rounds_config: ExtantFile, params_config: ExtantFile, images_folder: ExtantFolder) -> None:
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder)

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
    ]    
    logging.info(f"Running QC filtering of tracing supports: {' '.join(cmd_parts)}")
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
        str(H.proximity_filtered_spots_file_path),
        "-O", 
        H.analysis_path,
    ]
    logging.info(f"Running distance computation command: {' '.join(cmd_parts)}")
    return subprocess.check_call(cmd_parts)


def drift_correct_nuclei(rounds_config: ExtantFile, params_config: ExtantFile, images_folder: ExtantFolder) -> Path:
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder)
    N = NucDetector(H)
    return N.coarse_drift_correction_workflow()


def prep_locus_specific_spots_visualisation(rounds_config: ExtantFile, params_config: ExtantFile, images_folder: ExtantFolder) -> List[Path]:
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder)
    T = Tracer(H)
    per_fov_zarr = T.write_all_spot_images_to_one_per_fov_zarr()
    return per_fov_zarr


def _write_nuc_mask_table(*, fov: int, masks_table: pd.DataFrame, output_folder: Path) -> Path:
    fn = f"{get_position_name_short(fov)}.nuclear_masks.csv"
    fp = output_folder / fn
    logging.info(f"Writing data file for nuclei visualisation in FOV {fov}: {fp}")
    fp.parent.mkdir(exist_ok=True)
    masks_table.to_csv(fp)
    return fp


def prep_nuclear_masks_data(rounds_config: ExtantFile, params_config: ExtantFile, images_folder: ExtantFolder) -> Dict[int, Path]:
    """Write simple CSV data for visualisation of nuclear masks with napari plugin."""
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder)
    N = NucDetector(H)
    result: Dict[int, Path] = {}
    for fov, img in enumerate(N.mask_images):
        logging.info(f"Computing nuclear mask centroids for FOV: {fov}")
        table = extract_labeled_centroids(img)
        logging.info(f"Finished nuclear mask centroids for FOV: {fov}")
        result[fov] = _write_nuc_mask_table(fov=fov, masks_table=table, output_folder=H.nuclear_masks_visualisation_data_path)
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
            H.raw_spots_file, 
            H.proximity_filtered_spots_file_path, 
            H.nuclei_filtered_spots_file_path,
        ],
    )


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


class LooptracePipeline(pypiper.Pipeline):
    """Main looptrace processing pipeline"""

    def __init__(self, rounds_config: ExtantFile, params_config: ExtantFile, images_folder: ExtantFolder, output_folder: ExtantFolder, **pl_mgr_kwargs: Any) -> None:
        self.rounds_config = rounds_config
        self.params_config = params_config
        self.images_folder = images_folder
        super(LooptracePipeline, self).__init__(name=PIPE_NAME, outfolder=str(output_folder.path), **pl_mgr_kwargs)

    def stages(self) -> list[pypiper.Stage]:
        rounds_params_images = {"rounds_config": self.rounds_config, "params_config": self.params_config, "images_folder": self.images_folder}
        return [
            pypiper.Stage(name="imaging_rounds_validation", func=validate_imaging_rounds_config, f_kwargs={"rounds_config": self.rounds_config}),
            pypiper.Stage(name="pipeline_precheck", func=pretest, f_kwargs={"rounds_config": self.rounds_config, "params_config": self.params_config}),
            pypiper.Stage(name="zarr_production", func=run_zarr_production, f_kwargs=rounds_params_images),
            pypiper.Stage(name="nuclei_detection", func=run_nuclei_detection, f_kwargs=rounds_params_images),
            pypiper.Stage(name="nuclei_drift_correction", func=drift_correct_nuclei, f_kwargs=rounds_params_images),
            pypiper.Stage(name="nuclear_masks_visualisation_data_prep", func=prep_nuclear_masks_data, f_kwargs=rounds_params_images),
            pypiper.Stage(
                name="move_nuclear_masks_visualisation_data", 
                func=move_nuclear_masks_visualisation_data, 
                f_kwargs=rounds_params_images,
                nofail=True,
            ),
            pypiper.Stage(name="psf_extraction", func=run_psf_extraction, f_kwargs=rounds_params_images),
            # Really just for denoising, no need for structural disambiguation
            pypiper.Stage(name=DECON_STAGE_NAME, func=run_deconvolution, f_kwargs=rounds_params_images),
            pypiper.Stage(name="drift_correction__coarse", func=run_coarse_drift_correction, f_kwargs=rounds_params_images), 
            # Find/define all the bead ROIs in each (FOV, frame) pair.
            pypiper.Stage(name="bead_roi_generation", func=gen_all_bead_rois, f_kwargs=rounds_params_images),
            # Count detected bead ROIs for each timepoint, mainly to see if anything went awry during some phase of the imaging, e.g. air bubble.
            pypiper.Stage(name="bead_roi_detection_analysis", func=run_all_bead_roi_detection_analysis, f_kwargs=rounds_params_images),
            pypiper.Stage(name="bead_roi_partition", func=partition_bead_rois, f_kwargs=rounds_params_images),
            pypiper.Stage(name="drift_correction__fine", func=run_fine_drift_correction, f_kwargs=rounds_params_images),
            pypiper.Stage(name="drift_correction_accuracy_analysis", func=run_drift_correction_analysis, f_kwargs=rounds_params_images), 
            pypiper.Stage(
                name="drift_correction_accuracy_visualisation", 
                func=run_drift_correction_accuracy_visualisation, 
                f_kwargs={"params_config": self.params_config},
            ), 
            pypiper.Stage(name=SPOT_DETECTION_STAGE_NAME, func=run_spot_detection, f_kwargs=rounds_params_images), # generates *_rois.csv (regional spots)
            pypiper.Stage(name="spot_proximity_filtration", func=run_spot_proximity_filtration, f_kwargs=rounds_params_images),
            pypiper.Stage(name="spot_nucleus_filtration", func=run_spot_nucleus_filtration, f_kwargs=rounds_params_images), 
            pypiper.Stage(
                name="regional_spots_visualisation_data_prep", 
                func=run_regional_spot_viewing_prep, 
                f_kwargs={"rounds_config": self.rounds_config, "params_config": self.params_config},
                nofail=True,
            ),
            pypiper.Stage(
                name="spot_counts_visualisation__regional", 
                func=plot_spot_counts, 
                f_kwargs={"rounds_config": self.rounds_config, "params_config": self.params_config, "spot_type": SpotType.REGIONAL},
            ), 
            # computes pad_x_min, etc.; writes *_dc_rois.csv (much bigger, since regional spots x frames)
            pypiper.Stage(name="spot_bounding", func=run_spot_bounding, f_kwargs=rounds_params_images),
            pypiper.Stage(name="spot_extraction", func=run_spot_extraction, f_kwargs=rounds_params_images),
            pypiper.Stage(name="spot_zipping", func=run_spot_zipping, f_kwargs={"params_config": self.params_config, "images_folder": self.images_folder}),
            pypiper.Stage(name="tracing", func=run_chromatin_tracing, f_kwargs=rounds_params_images),
            pypiper.Stage(name="spot_region_distances", func=run_frame_name_and_distance_application, f_kwargs=rounds_params_images), 
            pypiper.Stage(name=TRACING_QC_STAGE_NAME, func=qc_locus_spots_and_prep_points, f_kwargs=rounds_params_images),
            pypiper.Stage(
                name="spot_counts_visualisation__locus_specific", 
                func=plot_spot_counts, 
                f_kwargs={"rounds_config": self.rounds_config, "params_config": self.params_config, "spot_type": SpotType.LOCUS_SPECIFIC},
            ), 
            pypiper.Stage(
                name="pairwise_distances__locus_specific", 
                func=compute_locus_pairwise_distances, 
                f_kwargs={"rounds_config": self.rounds_config, "params_config": self.params_config},
            ),
            pypiper.Stage(
                name="pairwise_distances__regional", 
                func=compute_region_pairwise_distances, 
                f_kwargs={"rounds_config": self.rounds_config, "params_config": self.params_config},
            ),
            pypiper.Stage(name="locus_specific_spots_visualisation_data_prep", func=prep_locus_specific_spots_visualisation, f_kwargs=rounds_params_images),

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
    logging.info(f"Building {PIPE_NAME} pipeline, using images from {opts.images_folder.path}")
    return LooptracePipeline(**kwargs)


def main(cmdl):
    opts = parse_cli(cmdl)
    pipeline = init(opts)
    logging.info("Running pipeline")
    pipeline.run(start_point=opts.start_point, stop_after=opts.stop_after)
    pipeline.wrapup()


if __name__ == "__main__":
    main(sys.argv[1:])
