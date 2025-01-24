"""Metadata about an individual trace"""

from dataclasses import dataclass
from itertools import tee
import json
from string import whitespace
from typing import Any, Iterable, Mapping, Optional, TypeAlias, TypeVar

import attrs
from expression import Option, Result, compose, curry_flip, fst, option, snd
from expression.collections import Seq
from expression.collections.seq import concat
from expression import result
from gertils.types import TimepointFrom0, TraceIdFrom0
from looptrace.utilities import find_counts_of_repeats, list_from_object, traverse_through_either, wrap_error_message, wrap_exception

Errors: TypeAlias = Seq[str]
_A = TypeVar("_A")
_T = TypeVar("_T", bound=type)


def _is_valid_trace_group_name(_1, _2, value: str) -> None:
    try:
        first, last = value[0], value[-1]
    except IndexError:
        raise ValueError("Trace group name cannot be empty")
    if first in whitespace or last in whitespace:
        raise ValueError("Trace group name can't start or end with whitespace")


def _is_valid_trace_group_name_option(_1, attr: attrs.Attribute, value: Any) -> None:
    match value:
        case option.Option(tag="none", none=_):
            pass # all good if we have an empty option
        case option.Option(tag="some", some=name):
            if not isinstance(name, TraceGroupName):
                raise TypeError(f"Option-wrapped value for attribute {attr.name} isn't a trace group name, but of type {type(name).__name__}")
        case _:
            raise TypeError(f"Value for attribute {attr.name} isn't an Option, but {type(value).__name__}")


@attrs.define(frozen=True, order=True)
class TraceGroupName:
    get = attrs.field(validator=[attrs.validators.instance_of(str), _is_valid_trace_group_name]) # type: str

    @classmethod
    def try_option(cls, s: str) -> Result[Option["TraceGroupName"], str]:
        match s:
            case "":
                return Result.Ok(Option.Nothing())
            case _:
                return cls.from_string(s).map(Option.Some)

    @classmethod
    def from_string(cls, s: str) -> Result["TraceGroupName", str]:
        try:
            name = cls(s)
        except (TypeError, ValueError) as e:
            return Result.Error(f"Failed to build {cls.__name__} from value ({s}): {e}")
        else:
            return Result.Ok(name)

    @classmethod
    def unsafe(cls, obj: Any) -> "TraceGroupName":
        match obj:
            case str():
                match cls.from_string(obj):
                    case result.Result(tag="ok", ok=name):
                        return name
                    case result.Result(tag="error", error=err_msg):
                        raise ValueError(f"Failed to parse trace group name: {err_msg}")
            case _:
                raise TypeError(f"Parse input for trace group name isn't str, but {type(obj).__name__}")


def trace_group_option_to_string(maybe_name: Option[TraceGroupName]) -> str:
    return maybe_name.map(lambda name: name.get).default_value("")


@curry_flip(1)
def _check_all_of_type(xs: Iterable[_A], t: _T) -> None:
    for x in tee(xs, 1)[0]:
        if not isinstance(x, t):
            raise TypeError(f"First item not of type {t.__name__} is of type {type(x).__name__}")


def _check_homogeneous(items: Iterable[_A], t: Optional[_T] = None) -> None:
    xs = tee(items, 1)[0] # Caution for if input is an iterator.
    if t is None:
        try:
            t = type(next(xs))
        except StopIteration:
            # Empty collection is trivially homogeneous.
            return
    _check_all_of_type(t)(xs)


@attrs.define(frozen=True)
class TraceGroupTimes:
    get = attrs.field(validator=[
        attrs.validators.instance_of(frozenset), 
        attrs.validators.min_len(2),
        lambda _1, _2, times: _check_homogeneous(times, TimepointFrom0),
        ]) # type: frozenset[TimepointFrom0]

    def __iter__(self) -> Iterable[TimepointFrom0]:
        return iter(self.get)

    @classmethod
    def from_list(cls, times: list[TimepointFrom0]) -> Result["TraceGroupTimes", str]:
        match list(find_counts_of_repeats(times)):
            case []:
                @wrap_error_message("Trace group from times list")
                @wrap_exception((TypeError, ValueError))
                def safe_build(ts: list[TimepointFrom0]) -> Result["TraceGroupTimes", str]:
                    return cls(frozenset(ts))
                
                return safe_build(times)
            case repeated:
                Result.Error(f"{len(repeated)} time(s) occurring more than once: {repeated}")


@attrs.define(frozen=True, kw_only=True)
class TraceGroup:
    name = attrs.field(validator=attrs.validators.instance_of(TraceGroupName)) # type: TraceGroupName
    times = attrs.field(validator=attrs.validators.instance_of(TraceGroupTimes)) # type: TraceGroupTimes


def _validate_trace_groups_content(groups: Iterable[TraceGroup]) -> None:
    repeat_names = list(find_counts_of_repeats(g.name for g in groups))
    if len(repeat_names) != 0:
        raise ValueError(f"Repeated name(s) among trace groups; counts: {repeat_names}")
    repeat_times = list(find_counts_of_repeats(t for g in groups for t in g.times))
    if len(repeat_times) > 0:
        raise ValueError(f"Repeated time(s) among trace groups; counts: {repeat_times}")


@attrs.define(frozen=True)
class PotentialTraceMetadata:
    groups = attrs.field(validator=[
        attrs.validators.instance_of(frozenset),
        attrs.validators.min_len(1),
        lambda _1, _2, values: _check_homogeneous(values, TraceGroup),
        lambda _1, _2, values: _validate_trace_groups_content(values),
    ]) # type: frozenset[TraceGroup]
    _times_by_group = attrs.field(init=False) # type: Mapping[TraceGroupName, frozenset[TimepointFrom0]]
    _trace_group_name_by_times = attrs.field(init=False) # type: Mapping[frozenset[TimepointFrom0], TraceGroupName]

    def __attrs_post_init__(self) -> None:
        # Here, finally, we establish the data structures to back our own code's desired queries.
        object.__setattr__(self, "_times_by_group", {g.name: g.times for g in self.groups})
        object.__setattr__(self, "_trace_group_name_by_times", {})
        for g in self.groups:
            ts = g.times
            try:
                name = self._trace_group_name_by_times[ts]
            except KeyError:
                self._trace_group_name_by_times[ts] = g.name
            else:
                raise ValueError(f"Already mapped times {ts} to name {name}; tried to re-map to {g.name}")

    def get_group_times(self, group: TraceGroupName) -> Option[TraceGroupTimes]:
        if not isinstance(group, TraceGroupName):
            raise TypeError(f"Query isn't a {TraceGroupName.__name__}, but a {type(group).__name__}")
        return Option.of_optional(self._times_by_group.get(group))

    @classmethod
    def from_mapping(cls, m: Mapping[str, object]) -> Result["PotentialTraceMetadata", list[str]]:
        def proc1(key: str, value: object) -> Result[TraceGroup, Errors]:
            name_result: Result[TraceGroupName, str] = read_trace_group_name(key)
            times_result: Result[TraceGroupTimes, str] = parse_trace_group_times(value)
            match name_result, times_result:
                case result.Result(tag="error", error=err_name), result.Result(tag="error", error=err_times):
                    return Result.Error(Seq.of(err_name, err_times))
                case _, _:
                    return name_result\
                        .map2(times_result, lambda name, times: TraceGroup(name=name, times=times))\
                        .map_error(Seq.of)
        
        def combine(state: Result[set[TraceGroup], Errors], new_result: Result[TraceGroup, Errors]) -> Result[set[TraceGroup], Errors]:
            match state, new_result:
                case result.Result(tag="error", error=old_messages), result.Result(tag="error", error=new_messages):
                    return Result.Error(Seq.of_iterable(concat(old_messages, new_messages)))
                case _, _:
                    return state.map2(new_result, lambda groups, new_group: {new_group, *groups})

        def step(acc: Result[set[TraceGroup], Errors], kv: tuple[str, object]) -> Result[set[TraceGroup, Errors]]:
            return combine(state=acc, new_result=proc1(*kv))

        return Seq.of_iterable(m.items())\
            .fold(step, Result.Ok(set()))\
            .map(compose(frozenset, cls))\
            .map_error(lambda errors: errors.to_list())


def _check_homogeneous_list(t: _T, *, attr_name: str, xs: Any) -> None:
    match xs:
        case list():
            if not _check_all_of_type(t)(xs):
                raise TypeError(f"Not all values for attribute '{attr_name}' are of type {t.__name__}")
        case _:
            raise TypeError(f"Value for attribute {attr_name} isn't list, but {type(xs).__name__}")


@attrs.define(frozen=True, kw_only=True)
class LocusSpotViewingReindexingDetermination:
    timepoints = attrs.field(validator=lambda _, attr, value: _check_homogeneous_list(TimepointFrom0, attr_name=attr.name, xs=value)) # type: list[TimepointFrom0]
    traces = attrs.field(validator=lambda _, attr, value: _check_homogeneous_list(int, attr_name=attr.name, xs=value)) # type: list[int]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Result["LocusSpotViewingReindexingDetermination", list[str]]:
        return traverse_through_either(_value_from_map(data))(["timepoints", "traces"])\
            .map_error(list)\
            .map(lambda args: cls(timepoints=[TimepointFrom0(t) for t in fst(args)], traces=snd(args)))

    @property
    def to_json(self) -> Mapping[str, list[int]]:
        return {"timepoints": [t.get for t in self.timepoints], "traces": self.traces}


@curry_flip(1)
def _value_from_map(k: str, m: Mapping[str, Any]) -> Result[Any, str]:
    return Option.of_optional(m.get(k)).to_result(f"Missing key '{k}'")


@attrs.define(frozen=True, order=True)
class LocusSpotViewingKey:
    field_of_view = attrs.field(validator=attrs.validators.instance_of(str)) # type: str
    trace_group_maybe = attrs.field(validator=_is_valid_trace_group_name_option) # type: Option[TraceGroupName]

    @classmethod
    def from_mapping(cls, m: Mapping[str, object]) -> Result["LocusSpotViewingKey", list[str]]:
        return traverse_through_either(_value_from_map(m))(["field_of_view", "trace_group_maybe"])\
            .map_error(list)\
            .bind(lambda args: TraceGroupName.try_option(snd(args)).map(lambda maybe_name: (fst(args), maybe_name)))\
            .map(lambda args: cls(field_of_view=fst(args), trace_group_maybe=snd(args)))

    @property
    def to_mapping(self) -> Mapping[str, str | Option[TraceGroup]]: # TODO: stricter, fixed-values subtype of Mapping here
        return {
            "field_of_view": self.field_of_view, 
            "trace_group_maybe": trace_group_option_to_string(self.trace_group_maybe), 
        }

    @property
    def to_string(self) -> str:
        fov_part: str = self.field_of_view
        trace_group_name_part: str = trace_group_option_to_string(self.trace_group_maybe)
        return f"{fov_part}{self._delimiter()}{trace_group_name_part}"

    @staticmethod
    def _delimiter() -> str:
        return "__"

    @classmethod
    def _split_raw(cls, s: str) -> Result[tuple[str, str], str]:
        try:
            lhs, rhs = s.split(cls._delimiter())
        except ValueError as e:
            return Result.Error(f"Failed to split raw value into 2 parts: {e}")
        else:
            return Result.Ok((lhs, rhs))

    @classmethod
    def from_string(cls, s: str) -> Result["LocusSpotViewingKey", str]:
        return cls._split_raw(s).bind(
            lambda pair: TraceGroupName.try_option(snd(pair)).map(lambda grp_opt: cls(fst(pair), grp_opt))
        )
    
    @classmethod
    def unsafe(cls, obj: Any) -> "LocusSpotViewingKey":
        match obj:
            case str():
                match cls.from_string(obj):
                    case result.Result(tag="ok", ok=key):
                        return key
                    case result.Result(tag="error", error=err_msg):
                        raise ValueError(f"Failed to parse {cls.__name__}: {err_msg}")
            case _:
                raise TypeError(f"Input to parse as {cls.__name__} isn't str, but {type(obj).__name__}")


@attrs.define(frozen=True, kw_only=True)
class RealizedTraceMetadata:
    traceId = attrs.field(validator=attrs.validators.instance_of(TraceIdFrom0)) # type: TraceIdFrom0
    group = attrs.field(validator=attrs.validators.instance_of(Option)) # type: Option[TraceGroup]


def parse_trace_group_times(data: object) -> Result[TraceGroupTimes, str]:
    return list_from_object(data)\
        .bind(lambda objs: traverse_through_either(timepoint_from_object)(objs)\
                .map_error(lambda msgs: f"{msgs.length} error(s) reading objects as timepoints: {'; '.join(msgs)}"))\
        .bind(TraceGroupTimes.from_list)


@wrap_error_message("Trace group name from string")
@wrap_exception((TypeError, ValueError))
def read_trace_group_name(s: str) -> Result[TraceGroupName, str]:
    return TraceGroupName(s)


@wrap_error_message("0-based timepoint from int")
@wrap_exception((TypeError, ValueError))
def timepoint_from_object(t: object) -> Result[TimepointFrom0, str]:
    return TimepointFrom0(t)
