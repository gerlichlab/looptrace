"""Analyze pixel value statistics in regions of interest, across channels."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
import json
import logging
from operator import itemgetter
from pathlib import Path
import sys
from typing import Callable, Iterable, Mapping, TypeAlias, TypeVar

from expression import Option, Result, compose, curry_flip, snd
from expression.collections import Seq, seq
from expression.core import option, result
from gertils import ExtantFile, ExtantFolder, compute_pixel_statistics
from gertils.geometry import ImagePoint3D
from gertils.types import ImagingChannel
import pandas as pd
from spotfishing.roi_tools import get_centroid_from_record

from looptrace import FIELD_OF_VIEW_COLUMN
from looptrace.Drifter import TIMEPOINT_COLUMN, X_PX_COARSE, Y_PX_COARSE, Z_PX_COARSE
from looptrace.ImageHandler import ImageHandler
from looptrace.NucDetector import NucDetector
from looptrace.utilities import find_first_option, get_either, list_from_object, traverse_through_either, wrap_exception, wrap_error_message

FieldOfViewName: TypeAlias = str
RawTimepoint: TypeAlias = int

_A = TypeVar("_A")
_K = TypeVar("_K")

SIGNAL_CHANNEL_COLUMN = "signalChannel"


def _parse_cmdl(cmdl: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze pixel value statistics in regions of interest, across channels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rounds-config", 
        required=True, 
        type=ExtantFile.from_string,
        help="Path to the imaging rounds configuration file",
    )
    parser.add_argument(
        "--params-config", 
        required=True, 
        type=ExtantFile.from_string,
        help="Path to the looptrace parameters configuration file",
    )
    parser.add_argument( # needed to build an ImageHandler with actual image files.
        "--images-folder", 
        required=True,
        type=ExtantFolder.from_string,
        help="Path to an experiment's images folder",
    )
    parser.add_argument(
        "--signal-config",
        type=ExtantFile.from_string,
        help="Path to the configuration file declaring ROI diameters and channels to analyse",
    )
    return parser.parse_args(cmdl)


class RoiType(Enum):
    """Based on ROI type, decide which pool of ROIs (by file path attribute on ImageHandler) to grab."""
    LocusSpecific = "traces_file_qc_filtered"
    Regional = "spots_for_voxels_definition_file"

    @property
    def file_attribute_on_image_handler(self) -> str:
        return self.value

    @classmethod
    def parse(cls, s: str) -> Option["RoiType"]:
        def eq(m: "RoiType") -> bool: return m.name.lower() == s.lower()
        return find_first_option(eq)(cls)

    @classmethod
    def from_string(cls, s: str) -> Option["RoiType"]:
        return cls.parse(s)
    
    @classmethod
    def from_object(cls, obj: object) -> Result["RoiType", str]:
        match obj:
            case str():
                return cls.from_string(obj).to_result(f"Not valid as a ROI type: {obj}")
            case _:
                return Result.Error(f"Object to parse as ROI type isn't string, but {type(obj).__name__}: {obj}")


@dataclass(kw_only=True, frozen=True)
class AnalyticalSpecification:
    roi_type: RoiType
    roi_diameter: int
    channels: set[ImagingChannel]

    def __post_init__(self) -> None:
        if self.roi_diameter <= 0:
            raise ValueError(f"ROI diameter must be strictly positive; got {self.roi_diameter}")

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> Result["AnalyticalSpecification", list[str]]:
        maybe_roi_type: Result[RoiType, Seq[str]] = \
            get_either(data)("roiType") \
                .bind(RoiType.from_object) \
                .map_error(lambda msg: Seq.of(msg))
        maybe_diameter: Result[int, Seq[str]] = \
            get_either(data)("roiDiameterInPixels") \
                .bind(_parse_roi_diameter) \
                .map_error(lambda msg: Seq.of(msg))
        maybe_channels: Result[set[ImagingChannel], Seq[str]] = \
            get_either(data)("channels") \
                .bind(list_from_object) \
                .map_error(Seq.of) \
                .bind(_parse_imaging_channels) \
                .bind(lambda channels: _ensure_unique(channels).map_error(Seq.of))
        match maybe_roi_type, maybe_diameter, maybe_channels:
            case result.Result(tag="ok", ok=roi_type), result.Result(tag="ok", ok=diameter), result.Result(tag="ok", ok=channels):
                return Result.Ok(cls(roi_type=roi_type, roi_diameter=diameter, channels=channels))
            case _:
                return Result.Error(list(seq.concat(
                    *Seq.of(maybe_roi_type, maybe_diameter, maybe_channels) \
                        .map(result.swap) \
                        .choose(result.to_option) # TODO: test performance tradeoff vs. compose(swap, to_option).
                )))


def workflow(
    *, 
    rounds_config: ExtantFile, 
    params_config: ExtantFile, 
    images_folder: ExtantFolder, 
    maybe_signal_config: Option[ExtantFile],
) -> None:
    match maybe_signal_config:
        case option.Option(tag="none", none=_):
            logging.info("No signal config file, skipping cross-channel signal analysis")
            return
        case option.Option(tag="some", some=signal_config_file):
            conf_path: Path = signal_config_file.path
            logging.info("Parsing signal analysis configuration: %s", conf_path)
            with conf_path.open(mode="r") as fh:
                conf_data = json.load(fh)
            if not isinstance(conf_data, list):
                raise TypeError(f"Parsed signal config data (from {conf_path}) is {type(conf_data).__name__}")
            match traverse_through_either(AnalyticalSpecification.from_mapping)(conf_data):
                case result.Result(tag="error", error=messages):
                    raise Exception(f"Failed to parse analytical specifications (from {conf_path}): {messages}")
                case result.Result(tag="ok", ok=analysis_specs):
                    H = ImageHandler(rounds_config=rounds_config, params_config=params_config, images_folder=images_folder)
                    N = NucDetector(H)

                    spot_drift_file: Path = H.drift_correction_file__coarse
                    logging.info("Reading nuclei drift file: %s", spot_drift_file)
                    all_spot_drifts: Mapping[tuple[FieldOfViewName, RawTimepoint], DriftRecord] = read_spot_drifts_file(spot_drift_file)

                    nuclei_drift_file: Path = H.nuclei_coarse_drift_correction_file
                    logging.info("Reading nuclei drift file: %s", nuclei_drift_file)
                    all_nuclei_drifts: Mapping[FieldOfViewName, DriftRecord] = read_signal_drifts_file(nuclei_drift_file)

                    for spec in analysis_specs:
                        # Get the ROIs of this type.
                        roi_type: RoiType = spec.roi_type
                        if roi_type == RoiType.LocusSpecific:
                            # TODO: implement
                            # TODO: discard regional spot timepoints from the bigger collection: https://github.com/gerlichlab/looptrace/issues/376
                            # TODO: will need to account for pixels vs. nanometers
                            # TODO: will need to account for different headers (e.g., z_px and z rather than zc, yc, etc.)
                            logging.error("Cross-channel analysis for locus-specific spots isn't yet supported, skipping!")
                            continue
                        logging.info("Analyzing signal for ROI type '%s'", roi_type.name)
                        rois_file: Path = getattr(H, roi_type.file_attribute_on_image_handler)
                        all_rois: pd.DataFrame = pd.read_csv(rois_file, index_col=False)
                        
                        # Build up the records for this ROI type, for all FOVs.
                        by_raw_channel: Mapping[int, list[dict]] = defaultdict(list)
                        # TODO: refactor with https://github.com/gerlichlab/gertils/issues/32
                        for fov, raw_img in N.iterate_over_pairs_of_fov_name_and_original_image():
                            fov: str = fov.removesuffix(".zarr")
                            img = raw_img.compute()
                            logging.info(f"Analysing signal for FOV: {fov}")
                            nuc_drift: DriftRecord = all_nuclei_drifts[fov]
                            rois: pd.DataFrame = all_rois[all_rois[FIELD_OF_VIEW_COLUMN] == fov]
                            logging.info("ROI count: %d", rois.shape[0])
                            for _, r in rois.iterrows():
                                timepoint: RawTimepoint = r[TIMEPOINT_COLUMN]
                                spot_drift: DriftRecord = all_spot_drifts[(fov, timepoint)]
                                pt0: ImagePoint3D = get_centroid_from_record(r)
                                try:
                                    dc_pt: ImagePoint3D = ImagePoint3D(
                                        z=pt0.z - nuc_drift.z + spot_drift.z, 
                                        y=pt0.y - nuc_drift.y + spot_drift.y, 
                                        x=pt0.x - nuc_drift.x + spot_drift.x, 
                                    )
                                except ValueError as e:
                                    logging.error(f"Can't compute shifted center for original center {pt0}: {e}")
                                else:
                                    if dc_pt.z < 0:
                                        logging.error(f"Can't extract signal for negative z-coordinate: {dc_pt.z}")
                                    else:
                                        try:
                                            stat_bundles = compute_pixel_statistics(
                                                img=img,
                                                pt=dc_pt,
                                                channels=spec.channels, 
                                                diameter=spec.roi_diameter,
                                                channel_column=SIGNAL_CHANNEL_COLUMN,
                                            )
                                        except ValueError as e:
                                            logging.error("Failed to extract signal for a point; error: %s", e)
                                        else:
                                            # TODO: need to be robust to bounding box with negative coordinate(s)
                                            # TODO: https://github.com/gerlichlab/gertils/issues/34
                                            for stats in stat_bundles:
                                                ch: int = stats[SIGNAL_CHANNEL_COLUMN]
                                                # Add the original record and signal stats to the growing collection for this channel.
                                                by_raw_channel[ch].append({**r.to_dict(), **stats})
                        
                        # Write the output file for this ROI type, across all FOVs.
                        for raw_channel, records in sorted(by_raw_channel.items(), key=itemgetter(0)):
                            stats_frame: pd.DataFrame = pd.DataFrame(records)
                            fn = f"signal_analysis__rois_{roi_type.name}__channel_{raw_channel}.csv"
                            outfile: Path = Path(H.analysis_path) / fn
                            logging.info("Writing output file: %s", outfile)
                            stats_frame.to_csv(outfile, index=False)


@dataclass(kw_only=True, frozen=True)
class DriftRecord:
    x: float
    y: float
    z: float


@curry_flip(1)
def read_drift_file(drift_file: Path, get_key: Callable[[pd.Series], _K]) -> Mapping[_K, "DriftRecord"]:
    drifts: pd.DataFrame = pd.read_csv(drift_file, index_col=False)
    colnames = {Z_PX_COARSE: "z", Y_PX_COARSE: "y", X_PX_COARSE: "x"}
    result: Mapping[_K, DriftRecord] = {}
    for key, rec in drifts.rename(columns=colnames).apply(
        lambda row: (get_key(row), DriftRecord(**row[colnames.values()].to_dict())), 
        axis=1,
    ):
        if key in result:
            raise ValueError(f"Repeated key in drift file ({drift_file}): {key}")
        result[key] = rec
    return result
    

read_signal_drifts_file: Callable[[Path], Mapping[FieldOfViewName, "DriftRecord"]] = \
    read_drift_file(lambda row: row[FIELD_OF_VIEW_COLUMN].removesuffix(".zarr"))

read_spot_drifts_file: Callable[[Path], Mapping[tuple[FieldOfViewName, RawTimepoint], "DriftRecord"]] = \
    read_drift_file(lambda row: (row[FIELD_OF_VIEW_COLUMN].removesuffix(".zarr"), row[TIMEPOINT_COLUMN]))



def _ensure_unique(items: Iterable[_A]) -> Result[set[_A], str]:
    try:
        counts = Counter(items)
    except TypeError as e:
        return Result.Error(f"Could not construct set of items: {e}")
    repeats: list[tuple[_A, int]] = Seq(counts.items()).filter(compose(snd, lambda n: n > 1)).to_list()
    return Result.Ok(set(counts.keys())) if not repeats else Result.Error(f"{len(repeats)} repeated item(s); counts: {repeats}")


def _parse_imaging_channels(channels: object) -> Result[list[ImagingChannel], Seq[str]]:
    return list_from_object(channels) \
        .map_error(lambda msg: Seq.of(msg)) \
        .bind(traverse_through_either(parse_imaging_channel)) \
        .map(lambda seq: seq.to_list())


def _parse_roi_diameter(obj: object) -> Result[int, str]:
    match obj:
        case int():
            return Result.Ok(obj)
        case _:
            return Result.Error(f"ROI diameter must be integer, not {type(obj).__name__} ({obj})")



@wrap_error_message("ImagingChannel from object")
@wrap_exception((TypeError, ValueError))
def parse_imaging_channel(obj: object) -> Result[ImagingChannel, str]:
    return ImagingChannel(obj)


def main(cmdl: list[str]) -> None:
    opts: argparse.Namespace = _parse_cmdl(cmdl)
    workflow(
        rounds_config=opts.rounds_config, 
        params_config=opts.params_config, 
        images_folder=opts.images_folder,
        maybe_signal_config=Option.of_obj(opts.signal_config),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
