"""Tests for naming integers according to value and semantics"""

from hypothesis import given, strategies as st
import pytest
from looptrace.integer_naming import *

__author__ = "Vince Reuter"


@given(st.integers(min_value=1, max_value=9997))
def test_get_position_name_N__length_is_number_given__when_legal(n):
    names = get_position_names_N(n)
    assert len(names) == n


@given(st.integers(min_value=1, max_value=9998))
def test_get_position_name_N_over_domain__is_equivalent_to_one_at_a_time__at_shared_domain(n):
    manual = [get_position_name_short(i) for i in range(n)]
    auto = get_position_names_N(n)
    assert manual == auto


@pytest.mark.parametrize(["pos", "exp"], [
    (5, "P0006"), 
    (9, "P0010"), (10, "P0011"),
    (19, "P0020"), (20, "P0021"), 
    (98, "P0099"), (99, "P0100"), 
    (100, "P0101"), (331, "P0332"), (999, "P1000"), 
    (1000, "P1001"), (4236, "P4237"),
    ])
def test_get_position_name_short_over_domain__is_accurate(pos, exp):
    assert get_position_name_short(pos) == exp


@pytest.mark.parametrize(
    ["fun", "arg", "expected_structure"], [
        (get_position_name_short, -1, ValueError("-1 is out-of-bounds [0, 9998] for for namer 'TenThousand'")), 
        (get_position_names_N, -1, ValueError("Number of names is negative: -1")),
        (get_position_name_short, 0, (lambda obs: obs, "P0001")), 
        (get_position_names_N, 0, (lambda obs: obs, [])), 
        (get_position_name_short, 9998, (lambda obs: obs, "P9999")), 
        (get_position_names_N, 9998, (len, 9998)), 
        (get_position_name_short, 9999, ValueError("9999 is out-of-bounds [0, 9998] for for namer 'TenThousand'")), 
        (get_position_names_N, 9999, (len, 9999)), 
        (get_position_name_short, 10000, ValueError("10000 is out-of-bounds [0, 9998] for for namer 'TenThousand'")), 
        (get_position_names_N, 10000, ValueError("9999 is out-of-bounds [0, 9998] for for namer 'TenThousand'")), 
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
    st.sampled_from((get_position_name_short, get_position_names_N)), 
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
def test_get_position_name__is_expected_error_when_input_is_not_int(func_num_pair):
    func, alleged_number = func_num_pair
    with pytest.raises(TypeError) as err_ctx:
        func(alleged_number)
    exp_msg = f"Index to name ({alleged_number}) (type={type(alleged_number).__name__}) is not integer-like!"
    assert str(err_ctx.value) == exp_msg
