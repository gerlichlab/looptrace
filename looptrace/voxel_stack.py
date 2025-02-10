"""Data types related to defining and using stacks of voxels, e.g. data from multiple timepoints for the same ROI"""

import logging
from typing import Any, Mapping, Union

import attrs
from expression import Option, Result, option, result
from gertils.types import FieldOfViewFrom1
import pandas as pd

from looptrace import FIELD_OF_VIEW_COLUMN
from looptrace.integer_naming import get_fov_name_short, parse_field_of_view_one_based_from_position_name_representation
from looptrace.trace_metadata import TraceGroupName, trace_group_option_to_string

NUMBER_OF_DIGITS_FOR_ROI_ID = 5

is_nonnegative = attrs.validators.ge(0)


def _is_valid_optional_trace_group(_, attribute: attrs.Attribute, value: Any):
    match value:
        case option.Option(tag="none", none=_):
            pass
        case option.Option(tag="some", some=maybe_trace_group):
            if not isinstance(maybe_trace_group, TraceGroupName):
                raise TypeError(
                    f"Value for {attribute.name} isn't option-wrapped trace group name, but {type(maybe_trace_group).__name__}"
                )
    if not isinstance(value, Option):
        raise TypeError(f"Value for {attribute.name} isn't Option, but {type(value).__name__}")
    

@attrs.define(frozen=True, kw_only=True)
class VoxelSize:
    z = attrs.field(validator=[
        attrs.validators.instance_of((int, float)),
        attrs.validators.gt(0)
    ]) # type: int | float
    y = attrs.field(validator=[
        attrs.validators.instance_of((int, float)),
        attrs.validators.gt(0)
    ]) # type: int | float
    x = attrs.field(validator=[
        attrs.validators.instance_of((int, float)),
        attrs.validators.gt(0)
    ]) # type: int | float

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> Result["VoxelSize", Exception]:
        try:
            return Result.Ok(cls.unsafe_from_mapping(m))
        except Exception as e:
            return Result.Error(e)

    @classmethod
    def unsafe_from_mapping(cls, m: Mapping[str, Any]) -> "VoxelSize":
        return cls(**m)

    @property
    def to_tuple(self) -> tuple[int | float, int | float, int | float]:
        return attrs.astuple(self)
    

@attrs.define(frozen=True, kw_only=True)
class VoxelStackSpecification:
    """
    Bundle of the essential provenance related to a particular stack of voxels.

    More specifically, we extract, for each detected regional spot / ROI, a voxel for each of a 
    number of timepoints from the course of the imaging experiment. For when these voxels stacks 
    are then used for tracing and for locus spot visualisation, we need to know these things:
    1. Field of view
    2. ROI ID
    3. Regional/reference timepoint (i.e., in which timepoint the spot was produced/detected)
    4. Trace group ID (if applicable, i.e. merging ROIs into a bigger tracing structure)
    5. Trace ID
    
    These data items can also be used to sort CSV-like records of locus spots, from which these 
    data may be extracted, as well as the voxels themseleves (e.g., when merging ROIs  for tracing, 
    and needing to put the voxels into a structure by aggregate trace ID for the visualisation).

    These data components also function as a key on which to join trace fits (i.e., parameters of 
    a 3D Gaussian) to the original locus spot ROI record, with things like bounding box information.

    Finally, note that while the trace ID should uniquely determine the value of the trace group, 
    we include this additional piece of data here so that we don't need to parse or otherwise 
    maintain a lookup table between trace ID and trace group name, or between the regional timepoints 
    (which comprise a trace group) and the trace group ID/name.
    """

    field_of_view = attrs.field(validator=attrs.validators.instance_of(FieldOfViewFrom1)) # type: FieldOfViewFrom1
    roiId = attrs.field(validator=[attrs.validators.instance_of(int), is_nonnegative]) # type: int
    ref_timepoint = attrs.field(validator=[attrs.validators.instance_of(int), is_nonnegative]) # type: int
    traceGroup = attrs.field(validator=_is_valid_optional_trace_group) # type: Option[TraceGroupName]
    traceId = attrs.field(validator=[attrs.validators.instance_of(int), is_nonnegative]) # type: int

    @property
    def file_name_base(self) -> str:
        return self._get_key_delimiter().join([
            get_fov_name_short(self.field_of_view), 
            str(self.roiId).zfill(NUMBER_OF_DIGITS_FOR_ROI_ID), 
            str(self.ref_timepoint),
            trace_group_option_to_string(self.traceGroup),
            str(self.traceId),
        ])

    @classmethod
    def from_file_name_base(cls, file_key: str) -> Result["VoxelStackSpecification", str]:
        try:
            raw_fov, roi, ref, raw_traceGroup, trace = file_key.split(cls._get_key_delimiter())
        except ValueError:
            logging.error(f"Failed to get key for file key: {file_key}")
            raise
        return cls._from_raw_values(
            raw_fov=raw_fov,
            raw_roiId=int(roi), 
            raw_ref_timepoint=int(ref),
            raw_traceGroup=raw_traceGroup,
            raw_traceId=int(trace),
        )
    
    @classmethod
    def from_file_name_base__unsafe(cls, file_key: str) -> "VoxelStackSpecification":
        match cls.from_file_name_base(file_key):
            case result.Result(tag="ok", ok=spec):
                return spec
            case result.Result(tag="error", error=err_msg):
                raise RuntimeError(f"Failed to build {cls.__name__} from given file name base; error: {err_msg}")

    @classmethod
    def from_roi_like(cls, roi: Union[pd.Series, Mapping[str, Any]]) -> Result["VoxelStackSpecification", str]:
        raw_traceGroup = roi["traceGroup"]
        if pd.isna(raw_traceGroup):
            raw_traceGroup = ""
        return cls._from_raw_values(
            raw_fov=roi[FIELD_OF_VIEW_COLUMN],
            raw_roiId=roi["roiId"], 
            raw_ref_timepoint=roi["ref_timepoint"],
            raw_traceGroup=raw_traceGroup,
            raw_traceId=roi["traceId"],
        )
    
    @classmethod
    def from_roi_like__unsafe(cls, roi: Union[pd.Series, Mapping[str, Any]]) -> "VoxelStackSpecification":
        match cls.from_roi_like(roi):
            case result.Result(tag="ok", ok=spec):
                return spec
            case result.Result(tag="error", error=err_msg):
                raise RuntimeError(err_msg)

    @property
    def name_roi_file(self) -> str:
        return self.file_name_base + ".npy"
    
    @staticmethod
    def row_order_columns() -> list[str]:
        """What's used to sort the rows of the all-voxel-specifications file, and the traces file."""
        # NB: we don't include the trace group ID here, as it's entirely determined by the ID. 
        return [FIELD_OF_VIEW_COLUMN, "roiId", "ref_timepoint", "traceId"]
    
    @classmethod
    def _from_raw_values(
        cls,
        *, 
        raw_fov: str, 
        raw_roiId: int, 
        raw_ref_timepoint: int, 
        raw_traceGroup: str, 
        raw_traceId: int,
    ) -> Result["VoxelStackSpecification", str]:
        return parse_field_of_view_one_based_from_position_name_representation(raw_fov.removesuffix(".zarr"))\
            .map(lambda fov: cls(
                field_of_view=fov, 
                roiId=raw_roiId, 
                ref_timepoint=raw_ref_timepoint,
                traceGroup=cls._build_trace_group(raw_traceGroup),
                traceId=raw_traceId,
            ))\
            .map_error(lambda err_msg: f"Failed to build {cls.__name__} instance from ROI -- {err_msg}")
    
    @staticmethod
    def _build_trace_group(raw_value: str) -> Option[TraceGroupName]:
        match TraceGroupName.try_option(raw_value):
            case result.Result(tag="ok", ok=maybe_name):
                return maybe_name
            case result.Result(tag="error", error=err_msg):
                raise ValueError(err_msg)

    @staticmethod
    def _get_key_delimiter() -> str:
        return "_"
