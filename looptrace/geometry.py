"""Various geometry abstractions and functions"""

import attrs


_int_or_float = attrs.validators.instance_of((int, float))


@attrs.define(kw_only=True, frozen=True)
class Point3D:
    """General abstraction of a point in 3D (assumed Euclidean) space"""
    x = attrs.field(validator=_int_or_float) # type: int | float
    y = attrs.field(validator=_int_or_float) # type: int | float
    z = attrs.field(validator=_int_or_float) # type: int | float
