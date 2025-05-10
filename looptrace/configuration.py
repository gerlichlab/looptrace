"""Tools related to looptrace configuration"""

from typing import Iterable, Literal, Mapping, TypeAlias
import yaml

from expression import Option, Result
from gertils import ExtantFile

from looptrace import ConfigurationValueError

__author__ = "Vince Reuter"
__credits__ = ["Vince Reuter"]


BACKGROUND_SUBTRACTION_TIMEPOINT_KEY = "subtract_background"
IMAGING_ROUNDS_KEY = "imagingRounds"
KEY_FOR_SEPARATION_NEEDED_TO_NOT_MERGE_ROIS = "distanceBeneathWhichSpotRoisWillMerge"
MAX_DISTANCE_SPOT_FROM_REGION_NAME = "max_dist"
SEMANTIC_KEY = "semantic"
TRACING_SUPPORT_EXCLUSIONS_KEY = "tracingExclusions"

ImageRound: TypeAlias = Mapping[str, object]


def determine_bead_timepoint_for_spot_filtration(
    *, 
    params_config: Mapping[str, object], 
    image_rounds: Iterable[ImageRound],
) -> Result[int, ConfigurationValueError]:
    return _get_bead_timepoint_for_spot_filtration(params_config)\
        .bind(lambda maybe_time: maybe_time.to_result("Could not get timepoint for filtration of spots by bead proximity"))\
        .bind(lambda t: _invalidate_beads_timepoint_for_spot_filtration(beads_timepoint=t, rounds=image_rounds))\
        .map_error(lambda msg: ConfigurationValueError(msg))


def get_minimum_regional_spot_separation(conf_data: Mapping[str, object]) -> int:
    """Get the minimum separation between regional spots as configured for a particular analysis / experiment."""
    section = get_region_grouping_config(conf_data)
    key = "minimumSeparation"
    try:
        return section[key]
    except KeyError as e:
        if section.get(SEMANTIC_KEY) == "UniversalProximityPermission":
            return 0
        raise ConfigurationValueError(
            f"Missing key ('{key}') for minimum separation for proximity-based filtration of regional spots"
        ) from e


def get_raw_bead_spot_proximity_filtration_threshold(config: Mapping[str, object]) -> Result[str, ConfigurationValueError]:
    key: Literal["beadSpotProximityDistanceThreshold"] = "beadSpotProximityDistanceThreshold"
    return Option.of_optional(config.get(key))\
        .to_result(f"Missing key ('{key}') for threshold for proximity-based filtration of spots by beads")\
        .map_error(ConfigurationValueError)


def get_region_grouping_config(conf_data: Mapping[str, object]) -> Mapping[str, object]:
    """Get the regional spots imaging configuration data for a particular analysis / experiment."""
    return conf_data["proximityFilterStrategy"]


def read_parameters_configuration_file(params_config: ExtantFile) -> Mapping[str, object]:
    """Parse the main looptrace parameters configuration file from YAML."""
    print(f"Reading looptrace parameters configuration file: {params_config.path}")
    with open(params_config.path, "r") as fh:
        return yaml.safe_load(fh)


def _get_bead_timepoint_for_spot_filtration(params_config: Mapping[str, object]) -> Result[Option[int], str]:
    bead_spot_filtration_key: Literal["proximityFiltrationBetweenBeadsAndSpots"] = "proximityFiltrationBetweenBeadsAndSpots"
    match params_config.get(bead_spot_filtration_key):
        case None:
            return Result.Error(f"Configuration is missing key for bead-to-spot proximity-based filtration: {bead_spot_filtration_key}")
        case False:
            return Result.Error(f"Bead-to-spot proximity-based filtration key ({bead_spot_filtration_key}) is set to False")
        case True:
            return Option\
                .of_optional(params_config.get(BACKGROUND_SUBTRACTION_TIMEPOINT_KEY))\
                .to_result(
                    f"Bead spot filtration key ({bead_spot_filtration_key}) is True, but backgrond subtraction timepoint key ({BACKGROUND_SUBTRACTION_TIMEPOINT_KEY}) is absent"
                )\
                .map(Option.Some)
        case int(t):
            return Result.Ok(Option.Some(t))
        case obj:
            return Result.Error(
                f"Bead-to-spot proximity-based filtration key ({bead_spot_filtration_key}) has value of illegal type: {type(obj).__name__}"
            )


def _invalidate_beads_timepoint_for_spot_filtration(
    *, 
    beads_timepoint: int, 
    rounds: Iterable[ImageRound],
) -> Result[int, str]:
    for r in rounds:
        if r["time"] == beads_timepoint:
            if r.get("isBlank", False):
                return Result.Ok(beads_timepoint)
            return Result.Error(f"The imaging round for beads timepoint {beads_timepoint} isn't tagged as being blank")
    return Result.Error(f"No imaging round corresponds to the given beads timepoint ({beads_timepoint})")
