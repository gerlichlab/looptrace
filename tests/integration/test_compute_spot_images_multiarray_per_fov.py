"""Tests for the computation of spot image visualisation data"""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
import json
from pathlib import Path
from unittest import mock

from gertils.types import TimepointFrom0, TraceIdFrom0
import hypothesis as hyp
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy
import hypothesis.extra.numpy as hyp_npy
import numpy as np
import pytest

from looptrace import ArrayDimensionalityError
from looptrace.ImageHandler import LocusGroupingData
from looptrace.SpotPicker import SPOT_IMAGE_PIXEL_VALUE_TYPE, RoiOrderingSpecification
from looptrace.Tracer import compute_spot_images_multiarray_per_fov
from looptrace.integer_naming import IntegerNaming, get_position_name_short


DEFAULT_MIN_NUM_PASSING = 500 # 5x the hypothesis default to ensure we really explore the search space
HYPOTHESIS_HEALTH_CHECK_SUPPRESSIONS = (hyp.HealthCheck.function_scoped_fixture, hyp.HealthCheck.too_slow, )
MIN_NUM_PASSING_FOR_SLOW_TEST_OF_ERROR_CASE = 10
MAX_RAW_FOV = IntegerNaming.TenThousand.value - 2 # Normal N digits accommodate 10^N - 1, and -1 more for 0-based
MAX_RAW_TRACE_ID = IntegerNaming.TenThousand.value - 2 # Normal N digits accommodate 10^N - 1, and -1 more for 0-based
NO_SHRINK_PHASES = tuple(p for p in hyp.Phase if p != hyp.Phase.shrink)

# Limit the data generation time and I/O time.
RATHER_SMALL_UPPER_BOUND_FOR_NUMBER_OF_DETECTED_SPOTS = 8
RATHER_SMALL_UPPER_BOUND_FOR_NUMBER_OF_LOCUS_TIMEPOINTS = 4
RATHER_SMALL_UPPER_BOUND_FOR_NUMBER_OF_REGIONAL_TIMEPOINTS = 3

gen_raw_fov = st.integers(min_value=0, max_value=MAX_RAW_FOV)

gen_spot_box = st.sampled_from([(2, 4, 4), (3, 3, 3)])

gen_trace_id = st.integers(min_value=0, max_value=MAX_RAW_TRACE_ID).map(TraceIdFrom0)

def get_name_for_raw_zero_based_fov(fov: int) -> str:
    return get_position_name_short(fov) + ".zarr"


BuildInput = tuple[Iterable[RoiOrderingSpecification.FilenameKey], LocusGroupingData]


@st.composite
def gen_legal_input(
    draw, 
    *, 
    max_num_fov: int = 3, 
    max_num_regional_times: int = 4, 
    min_num_locus_times_per_spot: int = 1, 
    max_num_locus_times_per_spot: int = RATHER_SMALL_UPPER_BOUND_FOR_NUMBER_OF_LOCUS_TIMEPOINTS, 
    allow_extras_in_locus_grouping: bool = True,
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
        RoiOrderingSpecification.FilenameKey(
            position=get_name_for_raw_zero_based_fov(fov),
            roi_id=tid.get,
            ref_frame=rt,
        ) 
        for (fov, rt), tid in zip(fov_rt_pairs, trace_ids, strict=True)
    ]

    # The regional timepoints actually used may not be the whole pool, so determine what we're really using.
    reg_times: set[TimepointFrom0] = {TimepointFrom0(k.ref_frame) for k in fn_keys}

    # Generate a number of locus timepoints for each regional timepoint (i.e., how many 3D volumes to generate for a spot, as a function of the spot's regional timepoint).
    num_loc_times_by_reg_time: dict[TimepointFrom0, int] = draw(st.lists(
        st.integers(min_value=min_num_locus_times_per_spot, max_value=max_num_locus_times_per_spot), 
        min_size=len(reg_times), 
        max_size=len(reg_times),
    ).map(lambda sizes: dict(zip(reg_times, sizes, strict=True))))

    # Generate the stack of spot image volumes for each regional spot, with the appropriate number of timepoints based on the regional spot identity.
    data = [
        # Add 1 to the number of locus times, to account for regional time itself.
        (k, draw(hyp_npy.arrays(dtype=SPOT_IMAGE_PIXEL_VALUE_TYPE, shape=(nt + 1, *spot_image_dims)))) 
        for k, nt in [
            (k, num_loc_times_by_reg_time[TimepointFrom0(k.ref_frame)]) 
            for k in sorted(fn_keys, key=lambda k: (k.position, k.roi_id, k.ref_frame))
        ]
    ]
    if not allow_empty_spots:
        assert len(data) > 0, "Empty spots data despite setting allow_empty_spots=False! "

    # Generate a dummy locus grouping.
    unused_reg_times: set[TimepointFrom0] = reg_times_pool - reg_times
    locus_grouping: dict[TimepointFrom0, set[TimepointFrom0]] = \
        {} \
        if not unused_reg_times or not allow_extras_in_locus_grouping \
        else (
            draw(st.dictionaries(
                keys=st.sampled_from(list(unused_reg_times)),
                values=st.sets(st.integers(min_value=0).map(TimepointFrom0).filter(lambda t: t not in reg_times_pool), max_size=5),
                max_size=len(unused_reg_times),
            ))
        )

    # Ensure that each regional timepoint for which data have been generated is present in the locus grouping, and with the appropriate number of locus timepoints.
    gen_locus_times: Callable[[int], SearchStrategy[set[TimepointFrom0]]] = \
        lambda n: st.sets(
            st.integers(min_value=0)
                .map(TimepointFrom0)
                .filter(lambda t: t not in reg_times_pool), min_size=n, max_size=n,
        )
    necessary_locus_grouping: dict[TimepointFrom0, set[TimepointFrom0]] = {rt: draw(gen_locus_times(n)) for rt, n in num_loc_times_by_reg_time.items()}
    locus_grouping.update(necessary_locus_grouping)
    if not allow_empty_locus_grouping:
        assert len(locus_grouping) > 0, "Empty locus grouping despite setting allow_empty_locus_grouping=False"
    final_locus_grouping = draw(st.one_of(st.just(locus_grouping), st.sampled_from((None, {})))) if allow_empty_locus_grouping else locus_grouping
    return data, final_locus_grouping


@hyp.given(fnkey_image_pairs_and_locus_grouping=gen_legal_input(allow_empty_spots=False))
@hyp.settings(
    max_examples=DEFAULT_MIN_NUM_PASSING, 
    phases=NO_SHRINK_PHASES,
    suppress_health_check=HYPOTHESIS_HEALTH_CHECK_SUPPRESSIONS,
)
def test_fields_of_view__are_correct_and_in_order(tmp_path, fnkey_image_pairs_and_locus_grouping):
    fnkey_image_pairs, locus_grouping = fnkey_image_pairs_and_locus_grouping
    npz_wrapper = mock_npz_wrapper(temp_folder=tmp_path, fnkey_image_pairs=fnkey_image_pairs)
    result = compute_spot_images_multiarray_per_fov(npz=npz_wrapper, locus_grouping=locus_grouping)
    obs = [fov_name for fov_name, _ in result]
    exp = list(sorted(set(k.position for k, _ in fnkey_image_pairs)))
    assert obs == exp


@hyp.given(fnkey_image_pairs_and_locus_grouping=gen_legal_input(allow_empty_spots=False))
@hyp.settings(
    max_examples=DEFAULT_MIN_NUM_PASSING, 
    phases=NO_SHRINK_PHASES,
    suppress_health_check=HYPOTHESIS_HEALTH_CHECK_SUPPRESSIONS,
)
def test_spot_images_finish_by_all_having_the_max_number_of_timepoints(tmp_path, fnkey_image_pairs_and_locus_grouping):
    fnkey_image_pairs, locus_grouping = fnkey_image_pairs_and_locus_grouping

    expected_num_locus_times_by_regional_time: Mapping[TimepointFrom0, int]
    if not locus_grouping:
        # In this case, there's no explicit grouping, so we just work with the data we're given. 
        # Then the expected number of locus timepoints per regional timepoint is simply what we observe.
        expected_num_locus_times_by_regional_time = get_locus_time_count_by_reg_time(fnkey_image_pairs)
    else:
        # Set the expected number of locus times per regional time by assuming correct relation between given grouping and the given data. 
        # This amounts to assuming that the data generating processes upstream of the system under test are correct, which is all well and good.
        # +1 to account for regional timepoint itself.
        expected_num_locus_times_by_regional_time = {rt: 1 + len(lts) for rt, lts in locus_grouping.items()}

    # For each FOV, determine which regional timepoints have spot data for that FOV.
    regional_times_by_fov: dict[str, list[TimepointFrom0]] = {}
    for fn_key, _ in fnkey_image_pairs:
        regional_times_by_fov.setdefault(fn_key.position, []).append(TimepointFrom0(fn_key.ref_frame))
    
    # Use the knowledge of regional times by FOV together with knowledge of number of locus times per regional time 
    # to determine the (max) number of locus times per FOV.
    expected_num_locus_times_by_fov: Mapping[str, int] = {
        fov_name: max(expected_num_locus_times_by_regional_time[rt] for rt in reg_times) 
        for fov_name, reg_times in regional_times_by_fov.items()
    }

    # Mock the input and make the call under test.
    npz_wrapper = mock_npz_wrapper(temp_folder=tmp_path, fnkey_image_pairs=fnkey_image_pairs)
    result: list[tuple[str, np.ndarray]] = compute_spot_images_multiarray_per_fov(
        npz=npz_wrapper, 
        locus_grouping=locus_grouping,
    )
    
    # Check that the time axis for each image stack for each FOV corresponds to that FOV's max number of locus times.
    observed_locus_times_by_fov: dict[str, int] = {fov_name: img.shape[1] for fov_name, img in result}
    assert observed_locus_times_by_fov == expected_num_locus_times_by_fov


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
    """If there's a regional timepoint for which we don't have expected locus time count, it's an error."""
    fnkey_image_pairs, locus_grouping = fnkey_image_pairs_and_locus_grouping
    npz_wrapper = mock_npz_wrapper(temp_folder=tmp_path, fnkey_image_pairs=fnkey_image_pairs)
    with pytest.raises(ArrayDimensionalityError) as error_context:
        compute_spot_images_multiarray_per_fov(npz=npz_wrapper, locus_grouping=locus_grouping)
    assert str(error_context.value).startswith("Locus times count doesn't match expectation")


@st.composite
def gen_input_with_missing_timepoint_counts(draw):
    # First make the initial (legal) input draw, and check that the data are nontrivial.
    fnkey_image_pairs, locus_grouping = draw(gen_legal_input(allow_empty_spots=False, allow_empty_locus_grouping=False, allow_extras_in_locus_grouping=False))
    real_regional_times: set[TimepointFrom0] = set(get_locus_time_count_by_reg_time(fnkey_image_pairs).keys())
    hyp.assume(len(set(locus_grouping.keys()).intersection(real_regional_times)) > 1) # need at least 2 in common so 1 can be dropped)
    new_locus_grouping = draw(st.lists(st.sampled_from(tuple(locus_grouping.items())), min_size=1, max_size=len(real_regional_times) - 1).map(dict))
    return fnkey_image_pairs, new_locus_grouping


@hyp.given(fnkey_image_pairs_and_locus_grouping=gen_input_with_missing_timepoint_counts())
@hyp.settings(
    max_examples=MIN_NUM_PASSING_FOR_SLOW_TEST_OF_ERROR_CASE,
    phases=NO_SHRINK_PHASES,
    suppress_health_check=HYPOTHESIS_HEALTH_CHECK_SUPPRESSIONS,
)
def test_regional_time_with_data_but_absent_from_nonempty_locus_grouping__causes_expected_error(tmp_path, fnkey_image_pairs_and_locus_grouping):
    fnkey_image_pairs, locus_grouping = fnkey_image_pairs_and_locus_grouping
    npz_wrapper = mock_npz_wrapper(temp_folder=tmp_path, fnkey_image_pairs=fnkey_image_pairs)
    with pytest.raises(RuntimeError) as error_context:
        compute_spot_images_multiarray_per_fov(npz=npz_wrapper, locus_grouping=locus_grouping)
    assert str(error_context.value).startswith("No expected locus time count for regional time")


def get_locus_time_count_by_reg_time(fnkey_image_pairs: Iterable[tuple[RoiOrderingSpecification.FilenameKey, np.ndarray]]) -> dict[TimepointFrom0, int]:
    result: dict[TimepointFrom0, int] = {}
    for key, img in fnkey_image_pairs:
        curr_num_time = img.shape[0]
        rt = TimepointFrom0(key.ref_frame)
        try:
            prev_num_time = result[rt]
        except KeyError:
            result[rt] = curr_num_time
        else:
            if prev_num_time != curr_num_time:
                raise RuntimeError(f"Had {prev_num_time} as number of locus timepoints for regional time {rt}, but then got {curr_num_time}")
    return result


def mock_npz_wrapper(*, temp_folder: Path, fnkey_image_pairs: Iterable[tuple[RoiOrderingSpecification.FilenameKey, np.ndarray]]):
    """Create a mocked version of the looptrace.image_io.NPZ_wrapper class, used to iterate over the locus spot images."""
    npz_wrapper = mock.Mock()
    npz_wrapper.npz = {k.file_name_base: img for k, img in fnkey_image_pairs}
    npz_wrapper.__iter__ = lambda self: iter(self.npz.values())
    npz_wrapper.files = list(npz_wrapper.npz.keys())
    npz_wrapper.__len__ = lambda self: sum(1 for _ in self)
    npz_wrapper.filepath = temp_folder / "dummy.npz" # used in error messages
    npz_wrapper.__getitem__ = lambda self, k: self.npz[k]
    return npz_wrapper
