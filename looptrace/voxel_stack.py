"""Data types related to defining and using stacks of voxels, e.g. data from multiple timepoints for the same ROI"""

import attrs
from expression import Option, Result, option, result
from typing import Any, Mapping, Union

import pandas as pd

from looptrace import FIELD_OF_VIEW_COLUMN
from looptrace.trace_metadata import TraceGroupName, trace_group_option_to_string

NUMBER_OF_DIGITS_FOR_ROI_ID = 5

is_nonnegative = attrs.validators.ge(0)


def _is_pos_int(_, attribute: attrs.Attribute, value: Any) -> None:
    if not isinstance(value, int):
        raise TypeError(f"Value for attribute {attribute.name} isn't int, but {type(value).__name__}")
    if value < 1:
        raise ValueError(f"Value for attribute {attribute.name} isn't positive: {value}")


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

    field_of_view = attrs.field(validator=attrs.validators.instance_of(str)) # type: str
    roiId = attrs.field(validator=[attrs.validators.instance_of(int), is_nonnegative]) # type: int
    ref_timepoint = attrs.field(validator=[attrs.validators.instance_of(int), is_nonnegative]) # type: int
    traceGroup = attrs.field(validator=_is_valid_optional_trace_group) # type: Option[TraceGroupName]
    traceId = attrs.field(validator=[attrs.validators.instance_of(int), is_nonnegative]) # type: int

    @property
    def file_name_base(self) -> str:
        return self._get_key_delimiter().join([
            self.field_of_view, 
            str(self.roiId).zfill(NUMBER_OF_DIGITS_FOR_ROI_ID), 
            str(self.ref_timepoint),
            trace_group_option_to_string(self.traceGroup),
            str(self.traceId),
        ])

    @classmethod
    def from_roi(cls, roi: Union[pd.Series, Mapping[str, Any]]) -> "VoxelStackSpecification":
        return cls(
            field_of_view=roi[FIELD_OF_VIEW_COLUMN], 
            roiId=roi["roiId"], 
            ref_timepoint=roi["ref_timepoint"],
            traceGroup=cls._build_trace_group(roi["traceGroup"]),
            traceId=roi["traceId"],
        )
    
    @classmethod
    def from_file_name_base(cls, file_key: str) -> "VoxelStackSpecification":
        try:
            fov, roi, ref, raw_trace_group, trace = file_key.split(cls._get_key_delimiter())
        except ValueError:
            print(f"Failed to get key for file key: {file_key}")
            raise
        return cls(
            field_of_view=fov, 
            roiId=int(roi), 
            ref_timepoint=int(ref),
            traceGroup=cls._build_trace_group(raw_trace_group),
            traceId=int(trace),
        )

    @property
    def name_roi_file(self) -> str:
        return self.file_name_base + ".npy"
    
    @staticmethod
    def row_order_columns() -> list[str]:
        """What's used to sort the rows of the all-voxel-specifications file, and the traces file."""
        # NB: we don't include the trace group ID here, as it's entirely determined by the ID. 
        return [FIELD_OF_VIEW_COLUMN, "roiId", "ref_timepoint", "traceId"]
    
    @staticmethod
    def _build_trace_group(raw_value: str) -> Option[TraceGroupName]:
        match TraceGroupName.try_option(raw_value):
            case result.Result(tag="ok", ok=maybe_name):
                return maybe_name
            case result.Result(tag="error", error=err_msg):
                raise ValueError(f"Could not build trace group name: {err_msg}")

    @staticmethod
    def _get_key_delimiter() -> str:
        return "_"
