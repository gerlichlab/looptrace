"""Tests for the PotentialTraceMetadata data type, which stores the merge rules for tracing"""

from typing import Callable, Iterable

from expression import Option, result
import pytest

from gertils.types import TimepointFrom0
from looptrace.trace_metadata import PotentialTraceMetadata, TraceGroup, TraceGroupName, TraceGroupTimes


def build_times(ts: Iterable[int]) -> TraceGroupTimes:
    return TraceGroupTimes(frozenset(map(TimepointFrom0, ts)))


GROUP_NAME: TraceGroupName = TraceGroupName("dummy")

TIMES: TraceGroupTimes = build_times([1, 2])

DUMMY_GROUP: TraceGroup = TraceGroup(name=GROUP_NAME, times=TIMES)


@pytest.mark.parametrize("wrap", [set, frozenset])
def test_groups_must_be_frozenset(wrap):
    groups = wrap([DUMMY_GROUP])
    if wrap == set:
        with pytest.raises(TypeError):
            PotentialTraceMetadata(groups)
    elif wrap == frozenset:
        md: PotentialTraceMetadata = PotentialTraceMetadata(groups)
        assert list(md.groups) == [DUMMY_GROUP]
    else:
        pytest.fail(f"Unexpected wrapper: {wrap}")
    

def test_groups_must_be_nonempty():
    with pytest.raises(ValueError):
        PotentialTraceMetadata(frozenset())


def test_group_name_repetition_is_prohibited():
    with pytest.raises(ValueError) as error_context:
        PotentialTraceMetadata(frozenset([
            TraceGroup(name=GROUP_NAME, times=build_times([1, 2])), 
            TraceGroup(name=GROUP_NAME, times=build_times([3, 4])),
        ]))
    assert "Repeated name(s) among trace groups" in str(error_context)


def test_timepoint_repetition_is_prohibited():
    with pytest.raises(ValueError) as error_context:
        PotentialTraceMetadata(frozenset([
            TraceGroup(name=TraceGroupName("a"), times=build_times([1, 2])), 
            TraceGroup(name=TraceGroupName("b"), times=build_times([2, 3])),
        ]))
    assert "Repeated time(s) among trace groups" in str(error_context)


@pytest.mark.parametrize(
    ["arg", "expected"], 
    [
        (TraceGroupName(GROUP_NAME.get.upper()), Option.Nothing()), 
        (GROUP_NAME, Option.Some(TIMES))
    ],
)
def test_get_group_times_is_correct(arg, expected):
    md: PotentialTraceMetadata = PotentialTraceMetadata(frozenset([DUMMY_GROUP]))
    assert md.get_group_times(arg) == expected


def test_roundtrip_through_mapping():
    lift_times: Callable[[list[int]], list[TimepointFrom0]] = lambda ts: [TimepointFrom0(t) for t in ts]
    a_times: list[int] = [1, 2]
    b_times: list[int] = [3, 4]
    match PotentialTraceMetadata.from_mapping({"A": a_times, "B": b_times}):
        case result.Result(tag="ok", ok=md):
            groups: list[TraceGroup] = list(sorted(md.groups, key=lambda g: g.name))
            assert [g.name.get for g in groups] == ["A", "B"]
            assert [list(sorted(g.times)) for g in groups] == [lift_times(a_times), lift_times(b_times)]
        case result.Result(tag="error", error=messages):
            pytest.fail(f"{len(messages)} problem(s) building potential trace metadata: {messages}")
