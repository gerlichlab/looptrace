"""Tests for the bundle of data which determines how to sort ROIs which are traced"""

from math import pow

from hypothesis import given, strategies as st

from looptrace.SpotPicker import NUMBER_OF_DIGITS_FOR_ROI_ID, RoiOrderingSpecification


DELIMITER = "_"
MAX_ROI_ID = int(pow(10, NUMBER_OF_DIGITS_FOR_ROI_ID)) - 1


def gen_regional_timepoint():
    return st.integers(min_value=0)


def gen_roi_id():
    return st.integers(min_value=0, max_value=MAX_ROI_ID)


def gen_string_with_no_delimiter():
    return st.text().filter(lambda s: DELIMITER not in s)


def gen_trace_id():
    return st.integers(min_value=0)


def gen_roi_ordering_specification():
    return st.tuples(
        gen_string_with_no_delimiter(),
        gen_trace_id(), 
        gen_roi_id(),
        gen_regional_timepoint(),
    ).map(lambda args: RoiOrderingSpecification(
        field_of_view=args[0], 
        traceId=args[1], 
        roiId=args[2], 
        ref_timepoint=args[3],
    ))


def gen_file_name_base():
    return st.tuples(
        gen_string_with_no_delimiter(), 
        gen_trace_id().map(str), 
        gen_roi_id().map(lambda tid: str(tid).zfill(NUMBER_OF_DIGITS_FOR_ROI_ID)), 
        gen_regional_timepoint().map(str),
    ).map(lambda components: DELIMITER.join(components))


@given(expected=gen_file_name_base())
def test_file_name_base_roundtrips_through_specification_instance(expected):
    """A valid file name base produces an ordering specification from which the original base is recovered."""
    ordering_specification = RoiOrderingSpecification.from_file_name_base(expected)
    observed = ordering_specification.file_name_base
    assert observed == expected


@given(expected=gen_roi_ordering_specification())
def test_specification_roundtrips_through_file_name_base(expected):
    """A valid ordering specification produces a file name base from which the original spec is recovered."""
    base = expected.file_name_base
    observed = RoiOrderingSpecification.from_file_name_base(base)
    assert observed == expected
