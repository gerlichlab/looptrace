"""Tests for the bundle of data which determines how to sort ROIs which are traced"""

from math import pow

from hypothesis import given, strategies as st
from hypothesis.strategies import SearchStrategy

from looptrace.trace_metadata import trace_group_option_to_string
from looptrace.voxel_stack import NUMBER_OF_DIGITS_FOR_ROI_ID, VoxelStackSpecification
from .utilities import gen_trace_group_maybe


DELIMITER = "_"
MAX_ROI_ID = int(pow(10, NUMBER_OF_DIGITS_FOR_ROI_ID)) - 1


def gen_regional_timepoint() -> SearchStrategy[int]:
    return st.integers(min_value=0)


def gen_roi_id() -> SearchStrategy[int]:
    return st.integers(min_value=0, max_value=MAX_ROI_ID)


def gen_string_with_no_delimiter() -> SearchStrategy[str]:
    return st.text().filter(lambda s: DELIMITER not in s)


def gen_trace_id() -> SearchStrategy[int]:
    return st.integers(min_value=0)


def gen_voxel_stack_specification() -> SearchStrategy[VoxelStackSpecification]:
    return st.tuples(
        gen_string_with_no_delimiter(),
        gen_roi_id(),
        gen_regional_timepoint(),
        gen_trace_group_maybe(),
        gen_trace_id(), 
    ).map(lambda args: VoxelStackSpecification(
        field_of_view=args[0], 
        roiId=args[1], 
        ref_timepoint=args[2],
        traceGroup=args[3],
        traceId=args[4], 
    ))


def gen_file_name_base() -> SearchStrategy[str]:
    return st.tuples(
        gen_string_with_no_delimiter(), 
        gen_roi_id().map(lambda tid: str(tid).zfill(NUMBER_OF_DIGITS_FOR_ROI_ID)), 
        gen_regional_timepoint().map(str),
        gen_trace_group_maybe().map(trace_group_option_to_string),
        gen_trace_id().map(str), 
    ).map(lambda components: DELIMITER.join(components))


@given(expected=gen_file_name_base())
def test_file_name_base_roundtrips_through_specification_instance(expected):
    """A valid file name base produces a specification from which the original base is recovered."""
    specification = VoxelStackSpecification.from_file_name_base(expected)
    observed = specification.file_name_base
    assert observed == expected


@given(expected=gen_voxel_stack_specification())
def test_specification_roundtrips_through_file_name_base(expected):
    """A valid specification produces a file name base from which the original spec is recovered."""
    base = expected.file_name_base
    observed = VoxelStackSpecification.from_file_name_base(base)
    assert observed == expected
