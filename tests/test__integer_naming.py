"""Tests for naming integers according to value and semantics"""

from math import log10
from expression import Result
from hypothesis import given, strategies as st
import pytest
from looptrace.integer_naming import IndexToNaturalNumberText, get_fov_names_N, get_fov_name_short

__author__ = "Vince Reuter"


@given(st.integers(min_value=1, max_value=9997))
def test_get_fov_name_N__length_is_number_given__when_legal(n):
    names = get_fov_names_N(n)
    assert len(names) == n


@given(st.integers(min_value=1, max_value=9998))
def test_get_fov_name_N_over_domain__is_equivalent_to_one_at_a_time__at_shared_domain(n):
    manual = [get_fov_name_short(i) for i in range(n)]
    auto = get_fov_names_N(n)
    assert manual == auto


@pytest.mark.parametrize(["pos", "exp"], [
    (5, "P0006"), 
    (9, "P0010"), (10, "P0011"),
    (19, "P0020"), (20, "P0021"), 
    (98, "P0099"), (99, "P0100"), 
    (100, "P0101"), (331, "P0332"), (999, "P1000"), 
    (1000, "P1001"), (4236, "P4237"),
    ])
def test_get_fov_name_short_over_domain__is_accurate(pos, exp):
    assert get_fov_name_short(pos) == exp


@pytest.mark.parametrize(
    ["fun", "arg", "expected_structure"], [
        (get_fov_name_short, -1, ValueError("-1 is out-of-bounds [0, 9999) for for namer 'TenThousand'")), 
        (get_fov_names_N, -1, ValueError("Number of names is negative: -1")),
        (get_fov_name_short, 0, (lambda obs: obs, "P0001")), 
        (get_fov_names_N, 0, (lambda obs: obs, [])), 
        (get_fov_name_short, 9998, (lambda obs: obs, "P9999")), 
        (get_fov_names_N, 9998, (len, 9998)), 
        (get_fov_name_short, 9999, ValueError("9999 is out-of-bounds [0, 9999) for for namer 'TenThousand'")), 
        (get_fov_names_N, 9999, (len, 9999)), 
        (get_fov_name_short, 10000, ValueError("10000 is out-of-bounds [0, 9999) for for namer 'TenThousand'")), 
        (get_fov_names_N, 10000, ValueError("9999 is out-of-bounds [0, 9999) for for namer 'TenThousand'")), 
    ])
def test_behavior_near_domain_boundaries(fun, arg, expected_structure):
    if isinstance(expected_structure, BaseException):
        with pytest.raises(type(expected_structure)) as obs_err:
            fun(arg)
        assert str(obs_err.value) == str(expected_structure)
    else:
        map_obs, exp = expected_structure
        raw_obs = fun(arg)
        obs = map_obs(raw_obs)
        assert obs == exp


@given(st.tuples(
    st.sampled_from((
        (get_fov_name_short, "Index to name"), 
        (get_fov_names_N, "Number of names"),
    )), 
    st.one_of(
        st.just(None),
        st.booleans(), 
        st.dates(), 
        st.datetimes(), 
        st.decimals(), 
        st.floats(), 
        st.lists(st.integers()), 
        st.sets(st.integers()), 
        st.tuples(st.integers()), 
        st.text()
        )
))
def test_get_fov_name__is_expected_error_when_input_is_not_int(func_and_prefix__num_pair):
    (func, exp_msg_prefix), alleged_number = func_and_prefix__num_pair
    with pytest.raises(TypeError) as err_ctx:
        func(alleged_number)
    exp_msg = f"{exp_msg_prefix} ({alleged_number}) (type={type(alleged_number).__name__}) is not integer-like!"
    assert str(err_ctx.value) == exp_msg


def get_legit_gen(namer: IndexToNaturalNumberText):
    # -1 to account for nonrepresentability of final value in domain, 
    # and -1 to account for the move from 0- to 1-based.
    return st.integers(min_value=0, max_value=namer.value - 2)


@st.composite
def gen_value_namer_pair(draw):
    namer = draw(st.from_type(IndexToNaturalNumberText))
    value = draw(get_legit_gen(namer))
    return value, namer


@given(value_namer_pair=gen_value_namer_pair())
def test_integers_roundtrip_through_string(value_namer_pair):
    value, namer = value_namer_pair
    expected = Result.Ok(value + 1)
    observed = namer.read_as_index(namer.get_name(value))
    assert observed == expected


@given(value_namer_pair=gen_value_namer_pair())
def test_number_of_characters_is_log10_of_namer_value(value_namer_pair):
    value, namer = value_namer_pair
    assert int(log10(namer.value)) == len(namer.get_name(value))
