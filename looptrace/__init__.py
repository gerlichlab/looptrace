"""Chromatin tracing in Python, from microscopy images of FISH probes"""

import pkg_resources
from gertils import ExtantFile

__all__ = ["LOOPTRACE_JAR_PATH"]


LOOPTRACE_JAR_PATH = ExtantFile.from_string(pkg_resources.resource_filename(__name__, "looptrace-assembly-0.1.0-SNAPSHOT.jar"))
