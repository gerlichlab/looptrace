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
from typing import Iterable, Mapping, TypeAlias, TypeVar

from expression import Option, Result, compose, snd
from expression.collections import Seq, seq
from expression.core import option, result
from gertils import ExtantFile, compute_pixel_statistics
from gertils.geometry import ImagePoint3D
from gertils.pixel_value_statistics import Numeric as PixelStatValue
from gertils.types import ImagingChannel
import numpy.typing as npt
import pandas as pd
from spotfishing.roi_tools import get_centroid_from_record

from looptrace import FIELD_OF_VIEW_COLUMN
from looptrace.Drifter import TIMEPOINT_COLUMN, X_PX_COARSE, Y_PX_COARSE, Z_PX_COARSE
from looptrace.ImageHandler import ImageHandler
from looptrace.SpotPicker import SpotPicker
from looptrace.utilities import find_first_option, get_either, wrap_exception, wrap_error_message

_A = TypeVar("_")

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
    parser.add_argument(
        "--signal-config",
        type=ExtantFile.from_string,
        help="Path to the configuration file declaring ROI diameters and channels to analyse",
    )
    return parser.parse_args(cmdl)


class RoiType(Enum):
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
                .bind(_list_from_object) \
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


def workflow(*, rounds_config: ExtantFile, params_config: ExtantFile, maybe_signal_config: Option[ExtantFile]) -> None:
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
                try:
                    analysis_specs: list[AnalyticalSpecification] = list(map(AnalyticalSpecification.from_mapping, conf_data))
                except (TypeError, ValueError) as e:
                    logging.error("Failed to parse analytical specifications (from %s): %s", conf_path, e)
                    raise

            H = ImageHandler(rounds_config=rounds_config, params_config=params_config)
            S = SpotPicker(H)

            spot_drift_file: Path = H.drift_correction_file__coarse
            logging.info("Reading nuclei drift file: %s", spot_drift_file)
            all_spot_drifts: pd.DataFrame = pd.read_csv(spot_drift_file, index_col=False)

            nuclei_drift_file: Path = H.nuclei_coarse_drift_correction_file
            logging.info("Reading nuclei drift file: %s", nuclei_drift_file)
            all_nuclei_drifts: pd.DataFrame = pd.read_csv(nuclei_drift_file, index_col=False)

            # TODO: discard regional spot timepoints from the bigger collection: https://github.com/gerlichlab/looptrace/issues/376
            for spec in analysis_specs:
                # Get the ROIs of this type.
                roi_type: RoiType = spec.roi_type
                logging.info("Analyzing signal for ROI type '%s'", roi_type.name)
                rois_file: Path = getattr(H, roi_type.file_attribute_on_image_handler)
                all_rois: pd.DataFrame = pd.read_csv(rois_file, index_col=False)
                
                # Build up the records for this ROI type, for all FOVs.
                by_raw_channel: Mapping[int, list[dict]] = defaultdict
                # TODO: refactor with https://github.com/gerlichlab/gertils/issues/32
                for fov, img in S.iter_fov_img_pairs():
                    logging.info(f"Analysing signal for FOV: {fov}")
                    nuc_drift_curr_fov: pd.DataFrame = all_nuclei_drifts[all_nuclei_drifts[FIELD_OF_VIEW_COLUMN] == fov]
                    logging.debug(f"Shape of nuclei drifts ({type(nuc_drift_curr_fov).__name__}): {nuc_drift_curr_fov.shape}")
                    spot_drifts_curr_fov: pd.DataFrame = all_spot_drifts[all_spot_drifts[FIELD_OF_VIEW_COLUMN] == fov]
                    logging.debug(f"Shape of spot drifts ({type(spot_drifts_curr_fov).__name__}): {spot_drifts_curr_fov.shape}")
                    rois: pd.DataFrame = all_rois[all_rois[FIELD_OF_VIEW_COLUMN] == fov]
                    logging.debug("ROI count: %d", rois.shape[0])
                    for _, r in rois.iterrows():
                        spot_drift = spot_drifts_curr_fov[spot_drifts_curr_fov[TIMEPOINT_COLUMN] == r[TIMEPOINT_COLUMN]]
                        pt0: ImagePoint3D = get_centroid_from_record(r)
                        dc_pt: ImagePoint3D = ImagePoint3D(
                            z=pt0.z - nuc_drift_curr_fov[Z_PX_COARSE] + spot_drift[Z_PX_COARSE], 
                            y=pt0.y - nuc_drift_curr_fov[Y_PX_COARSE] + spot_drift[Y_PX_COARSE], 
                            x=pt0.x - nuc_drift_curr_fov[X_PX_COARSE] + spot_drift[X_PX_COARSE], 
                        )
                        for stats in compute_pixel_statistics(
                            img=img,
                            pt=dc_pt,
                            channels=spec.channels, 
                            diameter=spec.roi_diameter,
                            channel_column=SIGNAL_CHANNEL_COLUMN,
                        ):
                            ch: int = stats[SIGNAL_CHANNEL_COLUMN]
                            by_raw_channel[ch].append({**r.to_dict(), **stats})
                
                # Write the output file for this ROI type, across all FOVs.
                for raw_channel, records in sorted(by_raw_channel.items(), key=itemgetter(0)):
                    stats_frame: pd.DataFrame = pd.DataFrame(records)
                    fn = f"signal_analysis__rois_{roi_type.name}__channel_{raw_channel}.csv"
                    outfile: Path = Path(H.analysis_path) / fn
                    logging.info("Writing output file: %s", outfile)
                    stats_frame.to_csv(outfile, index=False)
    

def _ensure_unique(items: Iterable[_A]) -> Result[set[_A], str]:
    try:
        counts = Counter(items)
    except TypeError as e:
        return Result.Error(f"Could not construct set of items: {e}")
    repeats: list[tuple[_A, int]] = Seq(counts.items()).filter(compose(snd, lambda n: n > 1)).to_list()
    return Result.Ok(set(counts.keys())) if not repeats else Result.Error(f"{len(repeats)} repeated item(s); counts: {repeats}")


def _parse_imaging_channels(channels: object) -> Result[list[ImagingChannel], Seq[str]]:
    State: TypeAlias = Result[Seq[ImagingChannel], Seq[str]]
    
    def proc1(acc: State, obj: object) -> State:
        match acc, _unsafe_parse_imaging_channel(obj):
            case result.Result(tag="ok", ok=goods), result.Result(tag="ok", ok=ch):
                return Result.Ok(goods.append(Seq.of(ch)))
            case result.Result(tag="ok", ok=_), result.Result(tag="error", error=err):
                return Result.Error(Seq.of(err)) # first error
            case result.Result(tag="error", error=bads), result.Result(tag="ok", ok=_):
                return Result.Error(bads)
            case result.Result(tag="error", error=bads), result.Result(tag="error", error=err):
                return Result.Error(bads.append(Seq.of(err)))

    return _list_from_object(channels) \
        .map_error(lambda msg: Seq.of(msg)) \
        .bind(lambda cs: Seq(cs).fold(proc1, Result.Ok(Seq()))) \
        .map(lambda seq: seq.to_list())


def _parse_roi_diameter(obj: object) -> Result[int, str]:
    match obj:
        case int():
            return Result.Ok(obj)
        case _:
            return Result.Error(f"ROI diameter must be integer, not {type(obj).__name__} ({obj})")


@wrap_error_message("Getting list from object")
@wrap_exception(TypeError)
def _list_from_object(obj: object) -> Result[list[object], str]:
    return list(obj)


@wrap_error_message("ImagingChannel from object")
@wrap_exception((TypeError, ValueError))
def _unsafe_parse_imaging_channel(obj: object) -> Result[ImagingChannel, str]:
    return ImagingChannel(obj)


def main(cmdl: list[str]) -> None:
    opts: argparse.Namespace = _parse_cmdl(cmdl)
    workflow(
        rounds_config=opts.rounds_config, 
        params_config=opts.params_config, 
        maybe_signal_config=Option.of_obj(opts.signal_config),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
