"""Tests for the computation of spot image visualisation data"""

from collections.abc import Iterable
from unittest import mock

from gertils.types import TimepointFrom0, TraceIdFrom0
import hypothesis as hyp
from hypothesis import strategies as st
import hypothesis.extra.numpy as hyp_npy
import pytest

from looptrace.SpotPicker import SPOT_IMAGE_PIXEL_VALUE_TYPE, RoiOrderingSpecification
from looptrace.Tracer import compute_spot_images_multiarray_per_fov
from looptrace.integer_naming import IntegerNaming, get_position_name_short

from tests.hypothesis_extra_strategies import gen_locus_grouping_data


# Limit the data generation time and I/O time.
RATHER_SMALL_UPPER_BOUND_FOR_NUMBER_OF_REGIONAL_TIMEPOINTS = 4
RATHER_SMALL_UPPER_BOUND_FOR_NUMBER_OF_DETECTED_SPOTS = 10

gen_raw_fov = st.integers(min_value=0, max_value=IntegerNaming.TenThousand.value)

gen_spot_img_dims = st.sampled_from([(2, 4, 4), (3, 3, 3)])

gen_trace_id = st.integers(min_value=0, max_value=IntegerNaming.TenThousand.value).map(TraceIdFrom0)

def get_name_for_raw_zero_based_fov(fov: int) -> str:
    return get_position_name_short(fov) + ".zarr"


@st.composite
def gen_legal_input(draw, *, max_num_fov: int = 3, max_num_regional_times: int = 4) -> tuple[Iterable[RoiOrderingSpecification.FilenameKey], dict[int, int]]:
    raw_reg_times_pool: set[int] = draw(st.sets(st.integers(min_value=0), min_size=1, max_size=max_num_regional_times))
    locus_grouping: dict[TimepointFrom0, list[TimepointFrom0]] = draw(gen_locus_grouping_data.with_strategies_and_empty_flag(
        gen_raw_reg_time=st.sampled_from(list(raw_reg_times_pool)),
        gen_raw_loc_time=st.integers(min_value=0).filter(lambda t: t not in raw_reg_times_pool),
        max_size=len(raw_reg_times_pool),
        allow_empty=True,
    ))
    locus_time_count_lookup: dict[int, int] = {} if len(locus_grouping) == 0 else draw(
        st.sets(st.sampled_from(list(locus_grouping.keys())))\
            .map(lambda rts: {t.get: len(locus_grouping[t]) for t in rts})
    )
    spot_image_dims = draw(gen_spot_img_dims)
    gen_spot_image = hyp_npy.arrays(dtype=SPOT_IMAGE_PIXEL_VALUE_TYPE, shape=spot_image_dims)
    images = draw(st.lists(gen_spot_image, max_size=RATHER_SMALL_UPPER_BOUND_FOR_NUMBER_OF_DETECTED_SPOTS))
    num_spots = len(images)
    raw_fov_pool: set[int] = draw(st.sets(gen_raw_fov, min_size=1, max_size=max_num_fov))
    fov_rt_pairs: set[tuple[int, int]] = draw(st.sets(
        st.tuples(
            st.sampled_from(list(raw_fov_pool)), 
            st.sampled_from(list(raw_reg_times_pool))
        ), 
        min_size=num_spots, 
        max_size=num_spots,
    ))
    trace_ids: set[TraceIdFrom0] = draw(st.sets(gen_trace_id, min_size=num_spots, max_size=num_spots))
    fn_keys = [
        RoiOrderingSpecification.FilenameKey(
            position=get_name_for_raw_zero_based_fov(fov),
            roi_id=tid.get,
            ref_frame=rt,
        ) 
        for (fov, rt), tid in zip(fov_rt_pairs, trace_ids, strict=True)
    ]
    return zip(sorted(fn_keys, key=lambda k: (k.position, k.roi_id, k.ref_frame)), images, strict=True), locus_time_count_lookup


@hyp.given(fnkey_image_pairs_and_loc_time_count_lookup=gen_legal_input())
@hyp.settings(
    phases=tuple(p for p in hyp.Phase if p != hyp.Phase.shrink),
    suppress_health_check=(hyp.HealthCheck.function_scoped_fixture, ),
)
def test_fields_of_view__are_correct_and_in_order(tmp_path, fnkey_image_pairs_and_loc_time_count_lookup):
    fnkey_image_pairs, loc_time_count_lookup = fnkey_image_pairs_and_loc_time_count_lookup
    npz_wrapper = mock.Mock()
    npz_wrapper.npz = {k.file_name_base: img for k, img in fnkey_image_pairs}
    npz_wrapper.files = list(npz_wrapper.npz.keys())
    npz_wrapper.filepath = tmp_path / "dummy.npz"
    npz_wrapper.__getitem__ = lambda _, k: npz_wrapper.npz[k]
    result = compute_spot_images_multiarray_per_fov(
        npz=npz_wrapper, 
        num_locus_times_per_regional_time=loc_time_count_lookup,
    )
    obs = [fov_name for fov_name, _ in result]
    exp = [get_name_for_raw_zero_based_fov(k.position) for k, _ in fnkey_image_pairs]
    assert obs == exp


@pytest.mark.skip("not implemented")
def test_regional_times_with_no_locus_times_are_ignored():
    """Expected count of 0, or absence of regional time from expectation mapping, should lead to regional time be ignored."""
    pass


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
