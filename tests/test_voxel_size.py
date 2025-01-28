"""Tests for the voxel size data type"""

import attrs
from expression import result
import hypothesis as hyp
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy
import pytest

from looptrace.voxel_stack import VoxelSize


def gen_valid_voxel_size() -> SearchStrategy[VoxelSize]:
    """Randomly generate a voxel size instance."""
    return st.lists(
        st.one_of(
            st.integers(min_value=1), 
            st.floats(min_value=0).filter(lambda x: x > 0),
        ), 
        min_size=3, 
        max_size=3,
    )\
    .map(lambda values: dict(zip(["z", "y", "x"], values, strict=True)))\
    .map(VoxelSize.unsafe_from_mapping)


@hyp.given(original_voxel_size=gen_valid_voxel_size())
def test_voxel_size_roundtrips_through_mapping(original_voxel_size):
    match VoxelSize.from_mapping(attrs.asdict(original_voxel_size)):
        case result.Result(tag="ok", ok=parsed_voxel_size):
            assert parsed_voxel_size == original_voxel_size
        case result.Result(tag="error", error=err_msg):
            pytest.fail(f"Failed to parse voxel size: {err_msg}")
        case unexpected:
            pytest.fail(f"Expected a Result-wrapped value but got a value of type {type(unexpected).__name__}")
