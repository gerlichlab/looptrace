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

from looptrace.ImageHandler import LocusGroupingData
from looptrace.SpotPicker import SPOT_IMAGE_PIXEL_VALUE_TYPE, RoiOrderingSpecification
from looptrace.Tracer import compute_spot_images_multiarray_per_fov
from looptrace.integer_naming import IntegerNaming, get_position_name_short

from tests.hypothesis_extra_strategies import gen_locus_grouping_data


DEFAULT_MIN_NUM_PASSING = 500 # 5x the hypothesis default to ensure we really explore the search space
MAX_RAW_FOV = IntegerNaming.TenThousand.value - 2 # Normal N digits accommodate 10^N - 1, and -1 more for 0-based
MAX_RAW_TRACE_ID = IntegerNaming.TenThousand.value - 2 # Normal N digits accommodate 10^N - 1, and -1 more for 0-based
NO_SHRINK_PHASES = tuple(p for p in hyp.Phase if p != hyp.Phase.shrink)

# Limit the data generation time and I/O time.
RATHER_SMALL_UPPER_BOUND_FOR_NUMBER_OF_DETECTED_SPOTS = 10
RATHER_SMALL_UPPER_BOUND_FOR_NUMBER_OF_LOCUS_TIMEPOINTS = 3
RATHER_SMALL_UPPER_BOUND_FOR_NUMBER_OF_REGIONAL_TIMEPOINTS = 4

gen_raw_fov = st.integers(min_value=0, max_value=MAX_RAW_FOV)

gen_spot_box = st.sampled_from([(2, 4, 4), (3, 3, 3)])

gen_trace_id = st.integers(min_value=0, max_value=MAX_RAW_TRACE_ID).map(TraceIdFrom0)

def get_name_for_raw_zero_based_fov(fov: int) -> str:
    return get_position_name_short(fov) + ".zarr"


@st.composite
def gen_legal_input(draw, *, max_num_fov: int = 3, max_num_regional_times: int = 4) -> tuple[Iterable[RoiOrderingSpecification.FilenameKey], LocusGroupingData]:
    # First, create a "pool" of regional timepoints to choose from in other generators.
    raw_reg_times_pool: set[int] = draw(st.sets(st.integers(min_value=0), min_size=1, max_size=max_num_regional_times))
    
    # Choose the size for each 3D spot image volume (needs to be constant).
    spot_image_dims = draw(gen_spot_box)
    
    # Choose how many regional spots to generate.
    num_spots = draw(st.integers(min_value=0, max_value=RATHER_SMALL_UPPER_BOUND_FOR_NUMBER_OF_DETECTED_SPOTS))
    
    # Get one or a few FOVs, as each regional spot image must be associated to a FOV.
    raw_fov_pool: set[int] = draw(st.sets(gen_raw_fov, min_size=1, max_size=max_num_fov))
    
    # Each spot comes from a particular FOV, and from a particular regional barcode. Draw these identifiers together.
    fov_rt_pairs: set[tuple[int, int]] = draw(st.sets(
        st.tuples(
            st.sampled_from(list(raw_fov_pool)), 
            st.sampled_from(list(raw_reg_times_pool))
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
        st.integers(min_value=1, max_value=RATHER_SMALL_UPPER_BOUND_FOR_NUMBER_OF_LOCUS_TIMEPOINTS), 
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

    # Generate a dummy locus grouping.
    locus_grouping: dict[TimepointFrom0, set[TimepointFrom0]] = draw(gen_locus_grouping_data.with_strategies_and_empty_flag(
        gen_raw_reg_time=st.sampled_from(list(raw_reg_times_pool)),
        gen_raw_loc_time=st.integers(min_value=0).filter(lambda t: t not in raw_reg_times_pool),
        max_size=len(raw_reg_times_pool),
        allow_empty=True,
    ))

    # Ensure that each regional timepoint for which data have been generated is present in the locus grouping, and with the appropriate number of locus timepoints.
    gen_locus_times: Callable[[int], SearchStrategy[set[TimepointFrom0]]] = lambda n: st.sets(
        st.integers(min_value=0).map(TimepointFrom0), min_size=n, max_size=n,
    ).filter(lambda lts: not any(t in reg_times for t in lts))
    necessary_locus_grouping: dict[TimepointFrom0, set[TimepointFrom0]] = {rt: draw(gen_locus_times(n)) for rt, n in num_loc_times_by_reg_time.items()}
    locus_grouping.update(necessary_locus_grouping)
    
    return data, locus_grouping


@hyp.given(fnkey_image_pairs_and_locus_grouping=gen_legal_input())
@hyp.settings(
    max_examples=DEFAULT_MIN_NUM_PASSING, 
    phases=NO_SHRINK_PHASES,
    suppress_health_check=(hyp.HealthCheck.function_scoped_fixture, ),
)
def test_fields_of_view__are_correct_and_in_order(tmp_path, fnkey_image_pairs_and_locus_grouping):
    fnkey_image_pairs, locus_grouping = fnkey_image_pairs_and_locus_grouping
    npz_wrapper = mock_npz_wrapper(temp_folder=tmp_path, fnkey_image_pairs=fnkey_image_pairs)

    # For nicer debugging
    print("LOCUS GROUPING (below)")
    print(json.dumps({rt.get: [lt.get for lt in lts] for rt, lts in locus_grouping.items()}, indent=2))

    result = compute_spot_images_multiarray_per_fov(
        npz=npz_wrapper, 
        locus_grouping=locus_grouping,
    )
    
    obs = [fov_name for fov_name, _ in result]
    exp = list(sorted(set(k.position for k, _ in fnkey_image_pairs)))
    assert obs == exp


@hyp.given(fnkey_image_pairs_and_locus_grouping=gen_legal_input())
@hyp.settings(
    max_examples=DEFAULT_MIN_NUM_PASSING, 
    phases=NO_SHRINK_PHASES,
    suppress_health_check=(hyp.HealthCheck.function_scoped_fixture, ),
)
def test_regional_times_with_no_locus_times_are_ignored(tmp_path, fnkey_image_pairs_and_locus_grouping):
    """Expected count of 0, or absence of regional time from expectation mapping, should lead to regional time be ignored."""
    fnkey_image_pairs, locus_grouping = fnkey_image_pairs_and_locus_grouping

    # For nicer debugging
    print("LOCUS GROUPING (below)")
    print(json.dumps({rt.get: [lt.get for lt in lts] for rt, lts in locus_grouping.items()}, indent=2))

    npz_wrapper = mock_npz_wrapper(temp_folder=tmp_path, fnkey_image_pairs=fnkey_image_pairs)

    # First, check that there are no regional timepoints (parsed from filename key) with data which are absent from the 
    # locus grouping, as this would represent an error in the creation of the NPZ file, which is upstream of what 
    # we're here endeavoring to test
    regional_times_histogram: Mapping[TimepointFrom0, int] = Counter(TimepointFrom0(k.ref_frame) for k, _ in fnkey_image_pairs)
    regional_times_with_data: set[TimepointFrom0] = set(regional_times_histogram.keys())
    regional_times_in_locus_grouping: set[TimepointFrom0] = set(locus_grouping.keys())
    missing_regional_times = regional_times_with_data - regional_times_in_locus_grouping
    extra_regional_times = regional_times_in_locus_grouping - regional_times_with_data
    assert missing_regional_times == set()
    # Check also that there is at least one regional time in the locus grouping which has no data associated with it, 
    # as this is the scenario for which we're trying here to test the behavior.
    hyp.assume(len(extra_regional_times) > 0)

    # Derive, from the input filename keys, the number of regional spots per FOV.
    exp_count_by_fov: Mapping[str, int] = Counter(k.position for k, _ in fnkey_image_pairs)

    # Compute the per-FOV image arrays.
    result: list[tuple[str, np.ndarray]] = compute_spot_images_multiarray_per_fov(
        npz=npz_wrapper, 
        locus_grouping=locus_grouping,
    )

    # Check that the number of stacks per FOV (first dimension of each resulting image) matches the expected number of regional timepoints per FOV.
    obs_count_by_fov = {fov_name: img.shape[0] for fov_name, img in result}
    assert all(fov_name in exp_count_by_fov for fov_name in set(obs_count_by_fov.keys()))
    
    # Check that each stack is 5-dimensional (ROI, time, z, y, x).
    observed_image_shapes = [img.shape for _, img in result]
    assert list(len(s) for s in observed_image_shapes) == [5] * len(observed_image_shapes)
    
    # Check that each regional spot's data was correctly stacked by field of view.
    assert obs_count_by_fov == exp_count_by_fov
    

@pytest.mark.skip("not implemented")
def test_spot_images_finish_by_being_all_the_same_size():
    pass


@pytest.mark.skip("not implemented")
def test_spot_images_finish_by_all_having_the_max_number_of_timepoints():
    pass


@pytest.mark.skip("not implemented")
def test_unexpected_timepoint_count_for_spot_image_volume__causes_expected_error():
    """If there's a regional timepoint for which we don't have expected locus time count, it's an error."""
    pass


@pytest.mark.skip("not implemented")
def test_spot_images_of_different_volume_sizes__causes_expected_error():
    """The bounding boxes are constructed to give uniform box/prism sizes; this must hold."""
    pass


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
