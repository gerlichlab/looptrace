"""Tools related to looptrace configuration"""

from typing import *
import yaml

from gertils import ExtantFile

__author__ = "Vince Reuter"
__credits__ = ["Vince Reuter"]

__all__ = [
    "MINIMUM_SPOT_SEPARATION_KEY", 
    "get_minimum_spot_separation", 
    "get_region_grouping_config",
    "read_parameters_configuration_file"
    ]


MINIMUM_SPOT_SEPARATION_KEY = "minimumPixelLikeSeparation"


def get_minimum_regional_spot_separation(conf_data) -> int:
    """Get the minimum separation between regional spots as configured for a particular analysis / experiment."""
    return get_region_grouping_config(conf_data)[MINIMUM_SPOT_SEPARATION_KEY]


def get_region_grouping_config(conf_data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Get the regional spots imaging configuration data for a particular analysis / experiment."""
    return conf_data["regionGrouping"]


def read_parameters_configuration_file(params_config: ExtantFile) -> Mapping[str, Any]:
    """Parse the main looptrace parameters configuration file from YAML."""
    print(f"Reading looptrace parameters configuration file: {params_config.path}")
    with open(params_config.path, "r") as fh:
        return yaml.safe_load(fh)
