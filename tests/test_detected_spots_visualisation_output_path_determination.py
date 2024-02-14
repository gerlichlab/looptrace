"""
Tests for the detected spots' images' output file path determination

The detected spot images (non-raw, semi-processed) are those which place ROI 
boxes around a centroid of a fit to a spot, and color the ROI box in the 2D 
image projection by the centroid's depth in z.
"""

import math
import string
import pytest
import hypothesis as hyp
from hypothesis import strategies as strat

from looptrace.SpotPicker import get_name_for_detected_spot_image_file

__author__ = "Vince Reuter"
__credits__ = ["Vince Reuter"]


# Control data random generation.
FIELD_DELIMITER = "_"
NUM_DIGITS_PER_FIELD = 3
MAXIMUM_REPRESENTABLE = int(math.pow(10, NUM_DIGITS_PER_FIELD)) - 1
PREFIX_ALPHABET = string.ascii_letters + string.digits + "_" + "-"
generate_representable_integer = strat.integers(min_value=0, max_value=MAXIMUM_REPRESENTABLE)
is_representable = lambda z: 0 <= z <= MAXIMUM_REPRESENTABLE
generate_nonrepresentable_integer = strat.integers().filter(lambda z: not is_representable(z))
representable_or_not = strat.one_of(generate_representable_integer, generate_nonrepresentable_integer)


@pytest.mark.parametrize(
    ["args", "kwargs", "exp_err"], [
        (args, kwargs, True) for args, kwargs in [
            (("Experiment", ), {"position": 1, "time": 2, "channel": 0}), 
            (("Experiment", 1), {"time": 2, "channel": 0}), 
            (("Experiment", 1, 2), {"channel": 0}), 
            (("Experiment", 1, 2, 0), {}),
        ]
    ] + [(tuple(), {"fn_prefix": "Experiment", "position": 1, "time": 2, "channel": 0}, False)]
)
def test_signature_requires_four_args_by_keyword(args, kwargs, exp_err):
    if exp_err:
        with pytest.raises(TypeError):
            get_name_for_detected_spot_image_file(*args, **kwargs)
    else:
        get_name_for_detected_spot_image_file(*args, **kwargs)


@hyp.given(
    prefix=strat.one_of(
        strat.text(set(PREFIX_ALPHABET) - {"", FIELD_DELIMITER}), 
        strat.just(""),
    ),
    position=generate_representable_integer,
    time=generate_representable_integer, 
    channel=generate_representable_integer,
)
def test_prefix_is_correctly_used(prefix, position, time, channel):
    fields = run_and_get_fields(fn_prefix=prefix, position=position, time=time, channel=channel)
    num_base_fields = 3
    if prefix == "":
        assert len(fields) == num_base_fields
    else:
        assert len(fields) == num_base_fields + prefix.count(FIELD_DELIMITER) + 1
        assert fields[0] == prefix


@hyp.given(
    prefix=strat.one_of(
        strat.text(set(PREFIX_ALPHABET) - {"", FIELD_DELIMITER}), 
        strat.just(""),
    ),
    position=generate_representable_integer,
    time=generate_representable_integer, 
    channel=generate_representable_integer,
    )
def test_three_fields_each_with_correct_prefix(prefix, position, time, channel):
    fields = run_and_get_fields(fn_prefix=prefix, position=position, time=time, channel=channel)
    fields_of_interest = fields[-3:] # The coding fields are at the end of the filename.
    assert len(fields_of_interest) * [4] == list(map(len, fields_of_interest)) # Each coding field has the correct size.
    assert ["P", "T", "C"] == [f[0] for f in fields_of_interest] # The prefixes are as expected.
    assert all([all(c in string.digits for c in f[1:]) for f in fields_of_interest]) # Every non-prefix character is a digit.


@hyp.given(
    args=strat.tuples(representable_or_not, representable_or_not, representable_or_not).filter(lambda args: not all(map(is_representable, args))), 
    prefix=strat.text(PREFIX_ALPHABET),
)
def test_out_of_bounds_values_give_expected_error(args, prefix):
    p, t, c = args
    with pytest.raises(ValueError):
        get_name_for_detected_spot_image_file(fn_prefix=prefix, position=p, time=t, channel=c)


@hyp.given(
    prefix=strat.text(PREFIX_ALPHABET),
    position=generate_representable_integer,
    time=generate_representable_integer, 
    channel=generate_representable_integer,
    filetype=strat.sampled_from(("png", "svg", "jpg", "jpeg", "bmp", "tiff", "tif")),
)
def test_suffix_is_correct_and_filetype_is_correctly_used(prefix, position, time, channel, filetype):
    fn = get_name_for_detected_spot_image_file(fn_prefix=prefix, position=position, time=time, channel=channel, filetype=filetype)
    suffix_fields = fn.split(".")
    content_semantic, obs_ext = suffix_fields[1:]
    assert content_semantic == "regional_spots" # The "extension" should have been prefix by the description of what's in the image file.
    assert obs_ext == filetype # The generated filename should have respected the filetype that was passed to it, giving a matching extension.


def run_and_get_fields(**kwargs) -> list[str]:
    fn = get_name_for_detected_spot_image_file(**kwargs)
    return fn.split(".")[0].split(FIELD_DELIMITER) # Handle possibility of underscore in part of file extension/suffix.
