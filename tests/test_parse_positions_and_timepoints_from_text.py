"""Tests for parsing field of view information from filename/path"""

from hypothesis import given, strategies as st
import pytest
from looptrace.image_io import POSITION_EXTRACTION_REGEX, TIME_EXTRACTION_REGEX, parse_fields_of_view_from_text, parse_times_from_text

__author__ = "Vince Reuter"


@pytest.mark.parametrize(
    ["parse", "filename", "expected"], 
    [pytest.param(parse, fn, exp, id=f"({parse.__name__}, {fn}, {exp})") for parse, fn, exp in 
        [
            (parse_fields_of_view_from_text, fn, exp) for fn, exp in [
                ("", []), 
                ("Does_Not_Contain_Keyword.nd2", []),
                ("My_Cool_File_T001_Point001.nd2", ["Point001"]),
                ("Point1Point2Point7.nd2", ["Point1", "Point2", "Point7"]), 
                ("My_Cool_File_Point00000010.nd2", ["Point00000010"]),
                ("Point00005.nd2", ["Point00005"]), 
                ("Point0Point1", ["Point0", "Point1"]), 
                ("T01", []), 
                ("My_Cool_File_Time001_P001.nd2", []),
                ("My_Cool_File_Time001_Point001.nd2", ["Point001"]),
                ("My_Cool_File_Time1_P001.nd2", []),
                ("My_Cool_File_T001_P001.nd2", []),
                ("My_Cool_File_T1_P001.nd2", []), 
                ("My_Cool_File_Time001_Point0001_Point0002_Point0006.nd2", ["Point0001", "Point0002", "Point0006"]),
                ]
        ] + 
        [
            (parse_times_from_text, fn, exp) for fn, exp in [
            ("", []), 
            ("Does_Not_Contain_Keyword.nd2", []),
            ("My_Cool_File_Time001_P001.nd2", ["Time001"]),
            ("Time1Time2Time7.nd2", ["Time1", "Time2", "Time7"]), 
            ("My_Cool_File_Time0010.nd2", ["Time0010"]),
            ("Time10.nd2", ["Time10"]), 
            ("Time0Time1", ["Time0", "Time1"]), 
            ("T01", []), 
            ("My_Cool_File_Tim001_P001.nd2", []),
            ("My_Cool_File_T001_P001.nd2", []),
            ("My_Cool_File_Time1_P001.nd2", ["Time1"]),
            ("My_Cool_File_T001_P001.nd2", []),
            ("My_Cool_File_T1_P001.nd2", []), 
            ("My_Cool_File_Time001_Point001_Time002_Point002.nd2", ["Time001", "Time002"]),
            ]
        ]
    ]
)
def test_accuracy(parse, filename, expected):
    """Affirm the accuracy of the parser on various inputs."""
    # This is an example-based test for accuracy.
    assert parse(filename) == expected


@st.composite
def good_regex_parse_pair(draw):
    pattern, parse = draw(st.sampled_from([
        (POSITION_EXTRACTION_REGEX, parse_fields_of_view_from_text), 
        (TIME_EXTRACTION_REGEX, parse_times_from_text),
        ]))
    filename = draw(st.from_regex(pattern))
    return parse, filename


@st.composite
def bad_regex_parse_pair(draw):
    pattern, parse = draw(st.sampled_from([
        (TIME_EXTRACTION_REGEX, parse_fields_of_view_from_text), 
        (POSITION_EXTRACTION_REGEX, parse_times_from_text),
        ]))
    filename = draw(st.from_regex(pattern))
    return parse, filename


@given(st.one_of(
    good_regex_parse_pair().map(lambda parse_fn_pair: parse_fn_pair + (True, )),
    st.one_of(
        bad_regex_parse_pair(),
        st.tuples(
            st.sampled_from((parse_fields_of_view_from_text, parse_times_from_text)),
            st.text().filter(lambda fn: "Point" not in fn and "Time" not in fn)
        ),
    ).map(lambda parse_fn_pair: parse_fn_pair + (False, ))
))
def test_any_hit_must_be_a_substring_of_input(parse__filename__exp_hit):
    """Assert universal property that any hit is a substring of input."""
    parse, filename, exp_hit = parse__filename__exp_hit
    result = parse(filename)
    if exp_hit:
        assert len(result) == 1
        assert result[0] in filename
    else:
        assert result == []
