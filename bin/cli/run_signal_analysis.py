"""Analyze pixel value statistics in regions of interest, across channels."""

import argparse
from collections import Counter
from dataclasses import dataclass
from enum import Enum
import logging
import sys
from typing import Iterable, Mapping, Optional, TypeAlias, TypeVar

from expression import Option, Result, compose, snd
from expression.collections import Seq, seq
from expression.core import result
from gertils import ExtantFile, compute_pixel_statistics
from gertils.types import ImagingChannel
import pandas as pd

from looptrace.ImageHandler import ImageHandler
from looptrace.utilities import find_first_option, get_either, wrap_exception, wrap_error_message

_A = TypeVar("_")


def _parse_cmdl(cmdl: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze pixel value statistics in regions of interest, across channels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument()
    return parser.parse_args(cmdl)


class RoiType(Enum):
    LocusSpecific = "traces_file_qc_filtered"
    Regional = "spots_for_voxels_definition_file"

    @property
    def image_handler_attribute(self) -> str:
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


def _ensure_unique(items: Iterable[_A]) -> Result[set[_A], str]:
    try:
        counts = Counter(items)
    except TypeError as e:
        return Result.Error(f"Could not construct set of items: {e}")
    repeats: list[tuple[_A, int]] = Seq(counts.items()).filter(compose(snd, lambda n: n > 1)).to_list()
    return Result.Ok(set(counts.keys())) if not repeats else Result.Error(f"{len(repeats)} repeated item(s); counts: {repeats}")


def _parse_roi_diameter(obj: object) -> Result[int, str]:
    match obj:
        case int():
            return Result.Ok(obj)
        case _:
            return Result.Error(f"ROI diameter must be integer, not {type(obj).__name__} ({obj})")


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


@wrap_error_message("Getting list from object")
@wrap_exception(TypeError)
def _list_from_object(obj: object) -> Result[list[object], str]:
    return list(obj)


@wrap_error_message("ImagingChannel from object")
@wrap_exception((TypeError, ValueError))
def _unsafe_parse_imaging_channel(obj: object) -> Result[ImagingChannel, str]:
    return ImagingChannel(obj)


def run_cross_channel_signal_analysis(
        rounds_config: ExtantFile, 
        params_config: ExtantFile, 
        signal_config: Optional[ExtantFile],
    ) -> None:
    if signal_config is None:
        logging.info("No signal config file, skipping cross-channel signal analysis")
        return
    H = ImageHandler(rounds_config=rounds_config, params_config=params_config)
    logging.info("Reading ROIs file: %s", str(H.region))
    rois = pd.read_csv(H.spots_for_voxels_definition_file, index_col=False)
    # TODO: apply drift correction (at least coarse for now)
    compute_pixel_statistics(
        # TODO: image
        # TODO: point
        channels=H.iter_signal_analysis_channels(), 
        diameter=H.signal_analysis_roi_diameter,
        channel_column="signalChannel",
    )
    # TODO: write to output files
    logging.info("Done with cross-channel signal analysis")


def workflow() -> None:
    pass


def main(cmdl: list[str]) -> None:
    opts: argparse.Namespace = _parse_cmdl(cmdl)


if __name__ == "__main__":
    main(sys.argv[1:])
