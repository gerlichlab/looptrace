"""Tests for the conversion of a stack of ND2 image files to dask array"""

from collections import OrderedDict, defaultdict
import string
from unittest import mock

import hypothesis as hyp
from hypothesis import strategies as st
import pytest

__author__ = "Vince Reuter"
__email__ = "vincent.reuter@imba.oeaw.ac.at"

from looptrace.nd2io import *
from looptrace.nd2io import AXIS_SIZES_KEY
from looptrace.integer_naming import get_fov_names_N

POSITION_PREFIX = "Point000"
non_neg_ints_pair = st.tuples(st.integers(min_value=0), st.integers(min_value=0))
uniq_non_neg_int_pair = non_neg_ints_pair.filter(lambda ab: ab[0] != ab[1])
zero_or_one = st.integers(min_value=0, max_value=1)
pos_time_zero_one = st.tuples(zero_or_one, zero_or_one)


def mocked_nd2_handle():
    """Mock object to imitate nd2.ND2File and its functionality used by functions under test"""
    handle = mock.Mock()
    handle.to_dask = lambda: []
    handle.__enter__ = lambda h: h
    handle.__exit__ = lambda _1, *_: None
    return handle


@pytest.mark.parametrize(
    "filenames", 
    [("img.tiff", ), ("img.tif", ), ("img.zarr", ), ("img.czi"), ("_img.nd2"), ()]
    )
def test_no_usable_images_raises_expected_error(tmp_path, filenames):
    paths = [tmp_path / fn for fn in filenames]
    for p in paths:
        p.touch()
    assert all(p.is_file for p in paths)
    with pytest.raises(EmptyImagesError):
        stack_nd2_to_dask(str(tmp_path))


@pytest.mark.parametrize(
    argnames="finalise_inputs", 
    argvalues=[pytest.param(f, id=name) for f, name in [ 
        (lambda _, fn: fn, "filename"), 
        (lambda p, fn: p / fn, "filepath"),
        (lambda p, fn: str(p / fn), "path_as_text"),
    ]])
@pytest.mark.parametrize("time_pos_flip", [False, True])
@hyp.given(
    fov_pair=uniq_non_neg_int_pair, 
    time_pair=uniq_non_neg_int_pair, 
    seq_num=st.integers(min_value=0),
    )
@hyp.settings(suppress_health_check=(hyp.HealthCheck.function_scoped_fixture, ))
def test_key_image_file_names_by_point_and_time__is_accurate(tmp_path, fov_pair, time_pair, seq_num, time_pos_flip, finalise_inputs):
    fov1, fov2 = fov_pair
    time1, time2 = time_pair
    args = []
    expected = defaultdict(dict)
    for i, (t, p) in enumerate(zip([f"Time000{t}" for t in [time1, time1, time2, time2]], [f"Point00{p}" for p in [fov1, fov2, fov1, fov2]])):
        extra = {"t": p, "p": t} if time_pos_flip else {"t": t, "p": p}
        arg = finalise_inputs(tmp_path, "{i}__{t}_{p}_ChannelFar Red,Red_Seq{sn}.nd2".format(**{**extra, **{"i": i, "sn": seq_num}}))
        expected[p][t] = str(arg)
        args.append(arg)
    assert key_image_file_names_by_point_and_time(args) == expected


@pytest.mark.parametrize(
    # Tuple bundling how to finalise argument to test function, 
    # how prepare the tempfolder, test function itself, and which input to choose.
    argnames="finalise_prepare_testfunc_choosearg_tuple", 
    argvalues=[pytest.param((f, p, test_func, choosearg), id=name) for f, p, test_func, choosearg, name in [ 
        (lambda _, fn: fn, lambda _1, _2: None, key_image_file_names_by_point_and_time, lambda _, fs: fs, "filename"), 
        (lambda p, fn: p / fn, lambda _1, _2: None, key_image_file_names_by_point_and_time, lambda _, fs: fs, "filepath"),
        (lambda p, fn: str(p / fn), lambda _1, _2: None, key_image_file_names_by_point_and_time, lambda _, fs: fs, "path_as_text"),
        (lambda _, fn: fn, lambda p, fns: [(p / fn).touch() for fn in fns], stack_nd2_to_dask, lambda p, _: p, "stack_nd2_to_dask")
    ]])
@pytest.mark.parametrize("time_pos_flip", [False, True])
@hyp.given(
    time_pos_pairs=st.lists(elements=st.sampled_from(((0,0), (0,1), (1,0), (1,1))), min_size=5, max_size=10).filter(lambda xs: len(xs) > len(set(xs))),
    seq_num=st.integers(min_value=0),
    )
@hyp.settings(
    # Avoid shrinking here since size range is narrow, so shrinking will yield little benefit and be slow.
    phases=tuple(p for p in hyp.Phase if p != hyp.Phase.shrink)
    )
def test_key_image_file_names_by_point_and_time__non_unique_fov_time_pairs_causes_expected_error(
    tmp_path_factory, seq_num, time_pos_pairs, time_pos_flip, finalise_prepare_testfunc_choosearg_tuple,
    ):
    finalise_args, prepare_folder, testfunc, choosearg = finalise_prepare_testfunc_choosearg_tuple
    filenames = [
        "{i}__Time0000{t}_Point000{p}_ChannelFar Red,Red_Seq{sn}.nd2".format(
            **{**({"t": p, "p": t} if time_pos_flip else {"t": t, "p": p}), **{"i": i, "sn": seq_num}}
            )
        for i, (t, p) in enumerate(time_pos_pairs)
        ]
    tmp_path = tmp_path_factory.mktemp("experiment")
    arg = choosearg(tmp_path, [finalise_args(tmp_path, fn) for fn in filenames])
    prepare_folder(tmp_path, filenames)
    with pytest.raises(FieldOfViewTimeFilenameKeyError):
        testfunc(arg)


@pytest.mark.parametrize("time_pos_flip", [False, True])
@hyp.given(
    fov_pair=uniq_non_neg_int_pair, 
    time_pair=uniq_non_neg_int_pair, 
    seq_num=st.integers(min_value=0),
    fov_index_choice=st.integers(min_value=4)
    )
@hyp.settings(
    # Avoid shrinking here since size range is narrow, so shrinking will yield little benefit and be slow.
    phases=tuple(p for p in hyp.Phase if p != hyp.Phase.shrink)
    )
def test_bad_fov_id__causes_expected_error(
    tmp_path_factory, fov_pair, time_pair, time_pos_flip, seq_num, fov_index_choice
    ):
    fov1, fov2 = fov_pair
    time1, time2 = time_pair
    filenames = [
        "{i}__Time000{t}_Point000{p}_ChannelFar Red,Red_Seq{sn}.nd2".format(
            **{**({"t": p, "p": t} if time_pos_flip else {"t": t, "p": p}), **{"i": i, "sn": seq_num}}
            )
        for i, (t, p) in enumerate(zip([time1, time1, time2, time2], [fov1, fov2, fov1, fov2]))
        ]
    tmp_path = tmp_path_factory.mktemp("experiment")
    for fn in filenames:
        (tmp_path / fn).touch()
    with pytest.raises(IndexError) as err_ctx:
        stack_nd2_to_dask(tmp_path, fov_index=fov_index_choice)
    exp_msg = f"{1 if fov1 == fov2 else 2} FOV name(s) available, but tried to select index {fov_index_choice}"
    assert str(err_ctx.value) == exp_msg


# Pair of filename extension and corresponding expected usage flag for file
pfx_ext_and_exp_use = st.one_of(
    st.just(("", "nd2", True)), # this one first to bias the strategy toward success
    st.tuples(
        st.sampled_from(("", "_")),
        st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=5), 
    ).map(lambda pfx_ext: (pfx_ext[0], pfx_ext[1], False)),
)


@hyp.given(input_params=st.lists(
    elements=st.tuples(
        non_neg_ints_pair, # (time, FOV)
        st.integers(min_value=0), # seq_num
        pfx_ext_and_exp_use,
        ),
    min_size=8, 
    max_size=16,
    unique_by=lambda t: t[0] # unique (time, FOV) pair for each filename-to-be
    ).filter(lambda params: any(par[2][2] is True for par in params))
)
@hyp.settings(
    # Avoid shrinking here since size range is narrow, so shrinking will yield little benefit and be slow.
    phases=tuple(p for p in hyp.Phase if p != hyp.Phase.shrink)
    )
def test_underscore_prefixed_and_or_non_nd2_files_are_skipped_and_good_ones_have_correct_fov_names(
    tmp_path_factory, input_params,
    ):
    """Result for list of FOV names must match expectation and correctly exclude the expect files to skip."""
    template = "{i}__Time000{t}_{p_pre}{p}_ChannelFar Red,Red_Seq{sn}.{ext}"
    unique_fields_of_view = set()
    exp_num_use = 0
    tmp_path = tmp_path_factory.mktemp("experiment")
    for i, ((t, p), seq_num, (pfx, ext, exp_use)) in enumerate(input_params):
        fn = pfx + template.format(i=i, t=t, p_pre=POSITION_PREFIX, p=p, sn=seq_num, ext=ext)
        (tmp_path / fn).touch()
        if exp_use:
            exp_num_use += 1
            unique_fields_of_view.add(p)
    # Patch the metadata parser to be a no-op, the ND2 reader to be context manager-like, 
    # and dask call to be identity.
    with mock.patch("looptrace.nd2io.parse_nd2_metadata", return_value={AXIS_SIZES_KEY: OrderedDict((dim, 0) for dim in ["Z", "C", "Y", "X"])}), \
        mock.patch("looptrace.nd2io.nd2.ND2File", side_effect=lambda *_, **__: mocked_nd2_handle()) as mock_nd2_read, \
        mock.patch("looptrace.nd2io.da.stack", side_effect=lambda arrs: arrs), \
        mock.patch("looptrace.nd2io.da.moveaxis", side_effect=lambda _1, _2, _3: mock.Mock(shape=None)):
        _, obs_pos_names, _ = stack_nd2_to_dask(tmp_path)
    assert len(mock_nd2_read.call_args_list) == exp_num_use
    assert obs_pos_names == get_fov_names_N(len(unique_fields_of_view))


@pytest.mark.parametrize(
    ["nd2_read_mocks", "get_err_prefix"], 
    [pytest.param(mm, get_pfx, id=name) for mm, get_pfx, name in [
        (
            [OSError(f"ND2 error!"), mocked_nd2_handle(), mocked_nd2_handle(), mocked_nd2_handle()], 
            lambda folder: f"Error reading first ND2 file from {folder}", 
            "beginning-of-stack",
        ), 
        (
            # Exactly 2 errors are expected, since the first and last 2 calls to the function 
            # under test are patched to not have an error, but the middle 2 are patched to raise error.
            [mocked_nd2_handle(), OSError(f"ND2 error!"), OSError(f"ND2 error!"), mocked_nd2_handle()], 
            lambda folder: f"2 error(s) reading ND2 files from {folder}", 
            "middle-of-stack",
        ),
    ]]
    )
@hyp.given(input_params=st.lists(
    elements=st.tuples(
        non_neg_ints_pair, # (time, FOV)
        st.integers(min_value=0), # seq_num
        ),
    min_size=4, 
    max_size=4,
    # unique FOV pair for each filename-to-be, so that pos_stack indexing isn't OOB
    unique_by=lambda t: t[0][1],
    )
)
@hyp.settings(
    # Avoid shrinking here since size range is narrow, so shrinking will yield little benefit and be slow.
    phases=tuple(p for p in hyp.Phase if p != hyp.Phase.shrink)
    )
def test_bad_nd2_files__causes_expected_error(
    tmp_path_factory, input_params, nd2_read_mocks, get_err_prefix
    ):
    """Error message type must match general expectation, and error message must match case-specific expectation."""
    template = "{i}__Time000{t}_{p_pre}{p}_ChannelFar Red,Red_Seq{sn}.nd2"
    filepaths = []
    tmp_path = tmp_path_factory.mktemp("experiment")
    for i, ((t, p), seq_num) in enumerate(input_params):
        fn = template.format(i=i, t=t, p_pre=POSITION_PREFIX, p=p, sn=seq_num)
        fp = (tmp_path / fn).touch()
        filepaths.append(fp)
    with pytest.raises(Nd2FileError) as err_ctx, \
        mock.patch("looptrace.nd2io.parse_nd2_metadata"), \
        mock.patch("looptrace.nd2io.nd2.ND2File", side_effect=nd2_read_mocks), \
        mock.patch("looptrace.nd2io.da.zeros_like", lambda _: [None]), \
        mock.patch("looptrace.nd2io.da.stack", side_effect=lambda x: [x]):
        stack_nd2_to_dask(tmp_path)
    exp_err_prefix = get_err_prefix(tmp_path)
    assert str(err_ctx.value).startswith(exp_err_prefix)
