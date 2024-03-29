"""Custom exception types to more accurately represent difficulties"""

__all__ = ["GpusUnavailableException", "MissingInputException", "MissingRoisTableException"]
__author__ = "Vince Reuter"


class GpusUnavailableException(Exception):
    """Error subtype for when GPUs must be available but aren't"""


class MissingInputException(Exception):
    """Error subtype for when some input is missing"""


class MissingRoisTableException(Exception):
    """Error subtype for when there's no ROIs table"""
