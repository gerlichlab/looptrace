"""Chromatin tracing in Python, from microscopy images of FISH probes"""

from pathlib import Path
from gertils import ExtantFile

__all__ = ["LOOPTRACE_JAR_PATH"]


LOOPTRACE_JAR_PATH = ExtantFile(Path(__file__).parent / "target"/ "scala-3.3.0" / "looptrace-assembly-0.1.0-SNAPSHOT.jar")
