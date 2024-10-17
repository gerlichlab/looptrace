"""Analyse bead discards by fail code/reason."""

import argparse
from collections import defaultdict
import json
import logging
from operator import itemgetter
from pathlib import Path
import sys
from typing import Optional

import pandas as pd
from gertils import ExtantFolder

from looptrace.ImageHandler import BeadRoisFilenameSpecification
from looptrace.bead_roi_generation import FAIL_CODE_COLUMN_NAME, INTENSITY_COLUMN_NAME, BeadRoiParameters


HISTOGRAM_FILENAME = "bead_rois_discard_analysis.json"
IMPOSSIBLES_FILENAME = "impossible_bead_cases.json"

AbsoluteMinimumShifting = 10
FailReasonHistogram = dict[BeadRoiParameters.BeadFailReason, int]
SingleResult = tuple[FailReasonHistogram, Optional[float]]


def parse_cmdl(cmdl: list[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyse bead discards by fail code/reason.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-I", "--input-folder", required=True, type=ExtantFolder.from_string, help="Path to folder in which to find files")
    parser.add_argument("-O", "--output-folder", required=True, type=Path, help="Path to folder in which to place output")
    parser.add_argument("--reference-timepoint", type=int, required=False, help="(0-based) timepoint of imaging sequence to be used as drift correction reference point")
    return parser.parse_args(cmdl)


def get_count_by_fail_code_and_get_max_min_intensity(table_with_codes: pd.DataFrame, *, min_record_count: int) -> SingleResult:
    known_codes: FailReasonHistogram = {code.value: code for code in BeadRoiParameters.BeadFailReason}
    if any(len(k) > 1 for k in known_codes):
        raise RuntimeError(f"Non-single-letter code(s): {', '.join(k for k in known_codes if len(k) > 1)}")
    count_by_code: dict[BeadRoiParameters.BeadFailReason, int] = defaultdict(int)
    for _, row in table_with_codes.iterrows():
        curr_codes = list(row[FAIL_CODE_COLUMN_NAME]) # Each code is single-letter.
        for code_value in curr_codes:
            try:
                code = known_codes[code_value]
            except KeyError as e:
                raise ValueError(f"Unknown code value ({code_value})! Known: {', '.join(known_codes)}") from e
            count_by_code[code] += 1
    good_or_only_failed_because_of_being_too_dim = \
        table_with_codes[table_with_codes[FAIL_CODE_COLUMN_NAME].isin(["", BeadRoiParameters.BeadFailReason.TooDim.value])]
    intensities = sorted(good_or_only_failed_because_of_being_too_dim[INTENSITY_COLUMN_NAME])
    max_min_intensity = None if len(intensities) < min_record_count else intensities[-min_record_count]
    return count_by_code, max_min_intensity


def workflow(*, root_folder: Path, output_folder: Path, reference_timepoint: Optional[int]) -> tuple[float, set[BeadRoisFilenameSpecification]]:
    logging.info("Finding and reading bead ROIs files in folder: %s...", root_folder)
    data_by_spec: dict[BeadRoisFilenameSpecification, SingleResult] = {}
    total_roi_count: int = 0
    for f in root_folder.iterdir():
        spec = None if not f.is_file() else BeadRoisFilenameSpecification.from_filepath(f)
        if spec in data_by_spec:
            raise RuntimeError(f"Bead ROIs filename spec parsed multiple times from filepaths in folder {root_folder}: {spec}")
        if spec is None:
            logging.debug("Bead ROIs filename spec not parsed from filepath %s, ignoring", f)
            continue
        assert spec.purpose is None, f"Non-null purpose for bead ROIs spec: {spec.purpose}"
        logging.debug("Parsing bead ROIs file %s (%s)...", f, str(spec))
        data = pd.read_csv(f, index_col=0)
        total_roi_count += data.shape[0]
        data[FAIL_CODE_COLUMN_NAME] = data[FAIL_CODE_COLUMN_NAME].fillna("")
        data_by_spec[spec] = get_count_by_fail_code_and_get_max_min_intensity(
            data, 
            min_record_count=AbsoluteMinimumShifting,
        )
    logging.info("Read %d bead ROI files...", len(data_by_spec))
    
    # Aggregate the individual data
    logging.info("Counting discarded ROIs by fail code...")
    count_by_fail: FailReasonHistogram = defaultdict(int)
    overall_max_min_intensity: Optional[float] = None
    impossible_specs: set[BeadRoisFilenameSpecification] = set()
    for spec, (histogram, curr_maxmin_intensity) in data_by_spec.items():
        if curr_maxmin_intensity is None:
            logging.debug(f"Impossible spec: {spec}")
            impossible_specs.add(spec)
        elif overall_max_min_intensity is None or curr_maxmin_intensity < overall_max_min_intensity:
            overall_max_min_intensity = curr_maxmin_intensity
            logging.debug(f"Updated maximum minimum intensity threshold: {curr_maxmin_intensity}")
        for code, count in histogram.items():
            count_by_fail[code] += count
    
    # Write the output file and print the maximum minimum intensity (to still get enough bead ROIs).
    logging.info("Unique failure reasons: %s", ", ".join(fail.name for fail in count_by_fail.keys()))
    output_folder.mkdir(exist_ok=True, parents=True)
    
    histogram_output_file = output_folder / HISTOGRAM_FILENAME
    logging.info("Writing fail reasons histogram file: %s", histogram_output_file)
    with histogram_output_file.open(mode="w") as fh:
        json.dump(dict(sorted(((k.name, v) for k, v in count_by_fail.items()), key=itemgetter(0))), fh, indent=2)
    
    logging.info("Impossible specs (listed below):")
    fovs_without_reference = []
    bad_times_by_fov: dict[int, list[int]] = {}
    for spec in sorted(impossible_specs):
        logging.info(spec)
        bad_times_by_fov.setdefault(spec.fov, []).append(spec.timepoint)
        if spec.timepoint == reference_timepoint:
            logging.warning("Impossible FOV for reference: %d", spec.fov)
            fovs_without_reference.append(spec.fov)
    impossibles_data = {
        "reference_timepoint_bead_impossibilities": fovs_without_reference,
        "general_bead_impossibilities": bad_times_by_fov
    }
    impossibles_output_file = output_folder / IMPOSSIBLES_FILENAME
    with impossibles_output_file.open(mode="w") as fh:
        json.dump(impossibles_data, fh, indent=2)
    
    logging.info(f"Maximum minimum intensity threshold: {overall_max_min_intensity}")
    logging.info("%d impossible specs (listed below)", len(impossible_specs))
    logging.info("Total spec count: %d", len(data_by_spec))
    logging.info("Total ROI count: %d", total_roi_count)

    return overall_max_min_intensity, impossible_specs


def main(cmdl: list[str]) -> None:
    logging.basicConfig(level=logging.DEBUG)
    opts = parse_cmdl(cmdl)
    workflow(
        root_folder=opts.input_folder.path, 
        output_folder=opts.output_folder,
        reference_timepoint=opts.reference_timepoint
    )

if __name__ == "__main__":
    main(sys.argv[1:])
