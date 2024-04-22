"""Regression tests for the configuration module"""

import pytest

from looptrace.configuration import SEMANTIC_KEY, get_minimum_regional_spot_separation

MAIN_SECTION_KEY = "proximityFilterStrategy"


@pytest.mark.parametrize(
    ("confdata", "exp_min_sep"), [
        ({MAIN_SECTION_KEY: {SEMANTIC_KEY: "UniversalProximityPermission"}}, 0), 
        ({MAIN_SECTION_KEY: {SEMANTIC_KEY: "UniversalProximityProhibition"}}, KeyError),
        ({MAIN_SECTION_KEY: {SEMANTIC_KEY: "SelectiveProximityPermission"}}, KeyError),
        ({MAIN_SECTION_KEY: {SEMANTIC_KEY: "SelectiveProximityProhibition"}}, KeyError),
    ]
)
def test_min_pixel_separation_is_required_if_and_only_if_not_in_universal_permissive_mode__issue_308(confdata, exp_min_sep):
    if isinstance(exp_min_sep, type) and issubclass(exp_min_sep, BaseException):
        with pytest.raises(exp_min_sep):
            get_minimum_regional_spot_separation(confdata)
    else:
        obs_min_sep = get_minimum_regional_spot_separation(confdata)
        assert obs_min_sep == exp_min_sep