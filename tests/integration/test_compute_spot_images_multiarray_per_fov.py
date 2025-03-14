"""Tests for the computation of spot image visualisation data"""

from collections.abc import Iterable
from pathlib import Path
from typing import Optional
from unittest import mock

from expression import Option
from gertils.types import FieldOfViewFrom1, TimepointFrom0, TraceIdFrom0
import hypothesis as hyp
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy
import hypothesis.extra.numpy as hyp_npy
import numpy as np
import pytest

from looptrace import ArrayDimensionalityError
from looptrace.ImageHandler import LocusGroupingData
from looptrace.SpotPicker import SPOT_IMAGE_PIXEL_VALUE_TYPE
from looptrace.Tracer import compute_locus_spot_voxel_stacks_for_visualisation
from looptrace.integer_naming import IndexToNaturalNumberText
from looptrace.voxel_stack import VoxelStackSpecification


DEFAULT_MIN_NUM_PASSING = 500 # 5x the hypothesis default to ensure we really explore the search space
HYPOTHESIS_HEALTH_CHECK_SUPPRESSIONS = (hyp.HealthCheck.function_scoped_fixture, hyp.HealthCheck.too_slow, )
MIN_NUM_PASSING_FOR_SLOW_TEST_OF_ERROR_CASE = 20
MAX_RAW_FOV = IndexToNaturalNumberText.TenThousand.value - 2 # Normal N digits accommodate 10^N - 1, and -1 more for 0-based
MAX_RAW_TRACE_ID = IndexToNaturalNumberText.TenThousand.value - 2 # Normal N digits accommodate 10^N - 1, and -1 more for 0-based
NO_SHRINK_PHASES = tuple(p for p in hyp.Phase if p != hyp.Phase.shrink)

# Limit the data generation time and I/O time.
RATHER_SMALL_UPPER_BOUND_FOR_NUMBER_OF_DETECTED_SPOTS = 8
RATHER_SMALL_UPPER_BOUND_FOR_NUMBER_OF_LOCUS_TIMEPOINTS = 4
RATHER_SMALL_UPPER_BOUND_FOR_NUMBER_OF_REGIONAL_TIMEPOINTS = 3

gen_raw_fov = st.integers(min_value=0, max_value=MAX_RAW_FOV)

gen_spot_box = st.sampled_from([(2, 4, 4), (3, 3, 3)])

gen_trace_id = st.integers(min_value=0, max_value=MAX_RAW_TRACE_ID).map(TraceIdFrom0)

BuildInput = tuple[Iterable[VoxelStackSpecification], LocusGroupingData]


def lift_zero_based_index_into_FieldOfViewFrom1(i: int) -> FieldOfViewFrom1:
    build_target = FieldOfViewFrom1
    if not isinstance(i, int):
        raise TypeError(f"Value to lift into {build_target.__name__} isn't int, but {type(i).__name__}")
    if i < 0:
        raise ValueError(f"Cannot lift negative value ({i}) into {build_target.__name__}")
    return build_target(i + 1)


@st.composite
def gen_legal_input(
    draw, 
    *, 
    max_num_fov: int = 3, 
    max_num_regional_times: int = 4, 
    min_num_locus_times_per_spot: int = 1, 
    max_num_locus_times_per_spot: int = RATHER_SMALL_UPPER_BOUND_FOR_NUMBER_OF_LOCUS_TIMEPOINTS, 
    allow_empty_spots: bool = True, 
    allow_empty_locus_grouping: bool = True,
) -> BuildInput:
    # First, create a "pool" of regional timepoints to choose from in other generators.
    reg_times_pool: set[TimepointFrom0] = draw(st.sets(st.integers(min_value=0).map(TimepointFrom0), min_size=max_num_regional_times, max_size=max_num_regional_times))
    
    # Choose the size for each 3D spot image volume (needs to be constant).
    spot_image_dims = draw(gen_spot_box)
    
    # Choose how many regional spots to generate.
    num_spots = draw(st.integers(min_value=0 if allow_empty_spots else 1, max_value=RATHER_SMALL_UPPER_BOUND_FOR_NUMBER_OF_DETECTED_SPOTS))
    
    # Get one or a few FOVs, as each regional spot image must be associated to a FOV.
    raw_fov_pool: set[int] = draw(st.sets(gen_raw_fov, min_size=1, max_size=max_num_fov))
    
    # Each spot comes from a particular FOV, and from a particular regional barcode. Draw these identifiers together.
    fov_rt_pairs: set[tuple[int, int]] = draw(st.sets(
        st.tuples(
            st.sampled_from(list(raw_fov_pool)), 
            st.sampled_from([rt.get for rt in reg_times_pool])
        ), 
        min_size=num_spots, 
        max_size=num_spots,
    ))

    # Randomise the trace IDs, creating a unique set of the appropriate size (each spot generates a trace ID).
    trace_ids: set[TraceIdFrom0] = draw(st.sets(gen_trace_id, min_size=num_spots, max_size=num_spots))
    
    # Create the Numpy array "filename" for each spot extraction 1 per eligible locus time, per regional spot.
    fn_keys = [
        VoxelStackSpecification(
            field_of_view=lift_zero_based_index_into_FieldOfViewFrom1(fov),
            roiId=tid.get, # For simplicity, just take the ROI ID as the trace ID.
            ref_timepoint=rt,
            traceGroup=Option.Nothing(),
            traceId=tid.get,
        ) 
        for (fov, rt), tid in zip(fov_rt_pairs, trace_ids, strict=True)
    ]

    # The regional timepoints actually used may not be the whole pool, so determine what we're really using.
    reg_times: set[TimepointFrom0] = {TimepointFrom0(k.ref_timepoint) for k in fn_keys}

    # Generate either a null or a fixed number of timepoints for the experiment, which will then be used 
    # to generate the primary dimension (time) of the spot image volume for each regional spot generated.
    gen_fixed_num_times = st.integers(min_value=len(reg_times), max_value=len(reg_times) + len(reg_times) * min_num_locus_times_per_spot)
    fixed_num_times: Optional[int] = (
        draw(gen_fixed_num_times) if allow_empty_locus_grouping else 
        draw(st.one_of(
            gen_fixed_num_times, 
            st.just(None),
        ))
    )

    # Generate a number of locus timepoints for each regional timepoint (i.e., how many 3D volumes 
    # to generate for a spot, as a function of the spot's regional timepoint).
    num_loc_times_by_reg_time: dict[TimepointFrom0, int] = (
        {rt: fixed_num_times for rt in reg_times} 
        if fixed_num_times is not None 
        else draw(
            st.lists(
                st.integers(min_value=min_num_locus_times_per_spot, max_value=max_num_locus_times_per_spot), 
                min_size=len(reg_times), 
                max_size=len(reg_times),
            ).map(lambda sizes: dict(zip(reg_times, sizes, strict=True)))
        )
    )

    # Generate the stack of spot image volumes for each regional spot, with the appropriate number of timepoints based on the regional spot identity.
    data = [
        # Add 1 to the number of locus times, to account for regional time itself.
        (k, draw(hyp_npy.arrays(dtype=SPOT_IMAGE_PIXEL_VALUE_TYPE, shape=(nt + 1, *spot_image_dims)))) 
        for k, nt in [
            (k, num_loc_times_by_reg_time[TimepointFrom0(k.ref_timepoint)]) 
            for k in sorted(fn_keys, key=lambda k: (k.field_of_view, k.roiId, k.ref_timepoint))
        ]
    ]
    if not allow_empty_spots:
        assert len(data) > 0, "Empty spots data despite setting allow_empty_spots=False! "

    # Ensure that each regional timepoint for which data have been generated is present in the locus grouping, and with the appropriate number of locus timepoints.
    def gen_locus_times(n: int) -> SearchStrategy[set[TimepointFrom0]]:
        return st.sets(
            st.integers(min_value=0)
                .map(TimepointFrom0)
                .filter(lambda t: t not in reg_times_pool), min_size=n, max_size=n,
        )
    def gen_locus_grouping():
        return st.just({rt: draw(gen_locus_times(n)) for rt, n in num_loc_times_by_reg_time.items()})
    locus_grouping: Optional[LocusGroupingData] = draw(
        st.one_of(gen_locus_grouping(), st.sampled_from(({}, None))) 
        if allow_empty_locus_grouping and fixed_num_times is not None 
        else gen_locus_grouping()
    )
    return data, locus_grouping


@hyp.given(fnkey_image_pairs_and_locus_grouping=gen_legal_input(allow_empty_spots=False))
@hyp.settings(
    max_examples=DEFAULT_MIN_NUM_PASSING, 
    phases=NO_SHRINK_PHASES,
    suppress_health_check=HYPOTHESIS_HEALTH_CHECK_SUPPRESSIONS,
)
def test_fields_of_view__are_correct_and_in_order(tmp_path, fnkey_image_pairs_and_locus_grouping):
    """The FOVs should be correctly parsed and sorted from the stack of extracted spot image volume files."""
    fnkey_image_pairs, locus_grouping = fnkey_image_pairs_and_locus_grouping
    npz_wrapper = mock_npz_wrapper(temp_folder=tmp_path, fnkey_image_pairs=fnkey_image_pairs)
    kwargs = {"locus_grouping": locus_grouping} if locus_grouping else {}
    result = compute_locus_spot_voxel_stacks_for_visualisation(
        npz=npz_wrapper, bg_npz=None, 
        num_timepoints=max(img.shape[0] for _, img in fnkey_image_pairs), 
        potential_trace_metadata=None, 
        **kwargs,
    )
    obs = [key.field_of_view for key, _ in result]
    exp = list(sorted(set(k.field_of_view for k, _ in fnkey_image_pairs)))
    assert obs == exp


@hyp.given(fnkey_image_pairs_and_locus_grouping=gen_legal_input(allow_empty_spots=False))
@hyp.settings(
    max_examples=DEFAULT_MIN_NUM_PASSING, 
    phases=NO_SHRINK_PHASES,
    suppress_health_check=HYPOTHESIS_HEALTH_CHECK_SUPPRESSIONS,
)
def test_spot_images_finish_by_all_having_the_max_number_of_timepoints(tmp_path, fnkey_image_pairs_and_locus_grouping):
    """Regardless of the locus grouping, each spot image volume should be padded out so that the time dimension is always the same."""
    fnkey_image_pairs, locus_grouping = fnkey_image_pairs_and_locus_grouping

    expected_num_timepoints: int
    if locus_grouping:
        # Set the expected number of locus times per regional time by assuming correct relation between given grouping and the given data. 
        # This amounts to assuming that the data generating processes upstream of the system under test are correct, which is all well and good.
        # +1 to account for regional timepoint itself.
        expected_num_timepoints = 1 + max(len(lts) for lts in locus_grouping.values())
    else:
        # In this case, there's no explicit grouping, so we just work with the data we're given. 
        # Then the expected number of locus timepoints per regional timepoint is simply what we observe.
        expected_num_timepoints = max(img.shape[0] for _, img in fnkey_image_pairs)
        
    # For each FOV, determine which regional timepoints have spot data for that FOV.
    regional_times_by_fov: dict[str, list[TimepointFrom0]] = {}
    for fn_key, _ in fnkey_image_pairs:
        regional_times_by_fov.setdefault(fn_key.field_of_view, []).append(TimepointFrom0(fn_key.ref_timepoint))

    # Mock the input and make the call under test.
    npz_wrapper = mock_npz_wrapper(temp_folder=tmp_path, fnkey_image_pairs=fnkey_image_pairs)
    kwargs = {"locus_grouping": locus_grouping} if locus_grouping else {}
    result = compute_locus_spot_voxel_stacks_for_visualisation(
        npz=npz_wrapper, 
        bg_npz=None, 
        num_timepoints=max(img.shape[0] for _, img in fnkey_image_pairs),
        potential_trace_metadata=None,
        **kwargs,
    )
    
    # Validate.
    try:
        next(iter(result))
    except StopIteration:
        pytest.fail("Empty result!")
    assert [expected_num_timepoints] * len(result) == [img.shape[2] for _, img in result]

@st.composite
def gen_input_with_bad_timepoint_counts(draw) -> BuildInput:
    # First make the initial (legal) input draw, and check that the data are nontrivial.
    fnkey_image_pairs, locus_grouping = draw(gen_legal_input(allow_empty_spots=False, allow_empty_locus_grouping=False, min_num_locus_times_per_spot=2))
    hyp.assume(locus_grouping is not None and len(locus_grouping) > 0 and len(fnkey_image_pairs) != 0)
    
    # Count up the locus times per regional time, and then create a new locus grouping in which at lease one 
    # regional time for which data were generated has fewer locus times in the grouping than it actually has in the data.
    real_locus_time_count_by_reg_time: dict[TimepointFrom0, int] = get_locus_time_count_by_reg_time(fnkey_image_pairs)
    original_locus_times = list(set(t for lts in locus_grouping.values() for t in lts))
    new_locus_grouping: dict[TimepointFrom0, set[TimepointFrom0]] = {rt: draw(st.sets(st.sampled_from(original_locus_times), min_size=1)) for rt in locus_grouping.keys()}
    # Check that indeed at least one regional time with data has fewer locus times in the new grouping than in the data.
    hyp.assume(any(len(locus_grouping[rt]) != len(new_locus_grouping[rt]) for rt in real_locus_time_count_by_reg_time.keys()))
    
    return fnkey_image_pairs, new_locus_grouping


@hyp.given(fnkey_image_pairs_and_locus_grouping=gen_input_with_bad_timepoint_counts())
@hyp.settings(
    max_examples=MIN_NUM_PASSING_FOR_SLOW_TEST_OF_ERROR_CASE,
    phases=NO_SHRINK_PHASES,
    suppress_health_check=HYPOTHESIS_HEALTH_CHECK_SUPPRESSIONS,
)
def test_unexpected_timepoint_count_for_spot_image_volume__causes_expected_error(tmp_path, fnkey_image_pairs_and_locus_grouping):
    """If there's a regional timepoint for which the expected timepoint count (before padding) differs from observation (before padding), it's an error."""
    fnkey_image_pairs, locus_grouping = fnkey_image_pairs_and_locus_grouping
    npz_wrapper = mock_npz_wrapper(temp_folder=tmp_path, fnkey_image_pairs=fnkey_image_pairs)
    with pytest.raises(ArrayDimensionalityError) as error_context:
        compute_locus_spot_voxel_stacks_for_visualisation(
            npz=npz_wrapper, 
            bg_npz=None, 
            num_timepoints=max(img.shape[0] for _, img in fnkey_image_pairs), 
            potential_trace_metadata=None, 
            locus_grouping=locus_grouping,
        )
    assert str(error_context.value).startswith("Timepoint count doesn't match expectation")


@st.composite
def gen_input_with_missing_timepoint_counts(draw):
    # First make the initial (legal) input draw, and check that the data are nontrivial.
    fnkey_image_pairs, locus_grouping = draw(gen_legal_input(allow_empty_spots=False, allow_empty_locus_grouping=False))
    real_regional_times: set[TimepointFrom0] = set(get_locus_time_count_by_reg_time(fnkey_image_pairs).keys())
    hyp.assume(len(set(locus_grouping.keys()).intersection(real_regional_times)) > 1) # need at least 2 in common so 1 can be dropped)
    new_locus_grouping = draw(st.lists(st.sampled_from(list(locus_grouping.items())), min_size=1, max_size=len(real_regional_times) - 1).map(dict))
    return fnkey_image_pairs, new_locus_grouping


@hyp.given(fnkey_image_pairs_and_locus_grouping=gen_input_with_missing_timepoint_counts())
@hyp.settings(
    max_examples=MIN_NUM_PASSING_FOR_SLOW_TEST_OF_ERROR_CASE,
    phases=NO_SHRINK_PHASES,
    suppress_health_check=HYPOTHESIS_HEALTH_CHECK_SUPPRESSIONS,
)
def test_regional_time_with_data_but_absent_from_nonempty_locus_grouping__causes_expected_error(tmp_path, fnkey_image_pairs_and_locus_grouping):
    """If there's a regional timepoint for which we don't have expected locus time count, it's an error."""
    fnkey_image_pairs, locus_grouping = fnkey_image_pairs_and_locus_grouping

    # for diagnostic aid in case of a failed test case
    import json
    print("locus grouping (below):")
    print(json.dumps({rt.get: [t.get for t in lts] for rt, lts in locus_grouping.items()}, indent=2))
    print("sizes (below)")
    print(json.dumps({k.name_roi_file: list(img.shape) for k, img in fnkey_image_pairs}, indent=2))
    
    npz_wrapper = mock_npz_wrapper(temp_folder=tmp_path, fnkey_image_pairs=fnkey_image_pairs)
    with pytest.raises(RuntimeError) as error_context:
        compute_locus_spot_voxel_stacks_for_visualisation(
            npz=npz_wrapper, 
            bg_npz=None, 
            num_timepoints=max(img.shape[0] for _, img in fnkey_image_pairs), 
            potential_trace_metadata=None, 
            locus_grouping=locus_grouping,
        )
    assert str(error_context.value).startswith("No expected locus time count for regional time")


def get_locus_time_count_by_reg_time(fnkey_image_pairs: Iterable[tuple[VoxelStackSpecification, np.ndarray]]) -> dict[TimepointFrom0, int]:
    """Get the number of timepoints from each image volume array, keying by regional timepoint and ensuring consensus among spots for each regional timepoint."""
    result: dict[TimepointFrom0, int] = {}
    for key, img in fnkey_image_pairs:
        curr_num_time = img.shape[0]
        rt = TimepointFrom0(key.ref_timepoint)
        try:
            prev_num_time = result[rt]
        except KeyError:
            result[rt] = curr_num_time
        else:
            if prev_num_time != curr_num_time:
                raise RuntimeError(f"Had {prev_num_time} as number of locus timepoints for regional time {rt}, but then got {curr_num_time}")
    return result


def mock_npz_wrapper(*, temp_folder: Path, fnkey_image_pairs: Iterable[tuple[VoxelStackSpecification, np.ndarray]]):
    """Create a mocked version of the looptrace.image_io.NPZ_wrapper class, used to iterate over the locus spot images."""
    npz_wrapper = mock.Mock()
    npz_wrapper.npz = {k.file_name_base: img for k, img in fnkey_image_pairs}
    npz_wrapper.__iter__ = lambda self: iter(self.npz.values())
    npz_wrapper.files = list(npz_wrapper.npz.keys())
    npz_wrapper.__len__ = lambda self: sum(1 for _ in self)
    npz_wrapper.filepath = temp_folder / "dummy.npz" # used in error messages
    npz_wrapper.__getitem__ = lambda self, k: self.npz[k]
    return npz_wrapper
