"""Tools for working with paths"""

from dataclasses import dataclass
from pathlib import Path

__all__ = ["ExtantFile", "ExtantFolder"]


@dataclass
class ExtantFile:
    path: Path

    def __post_init__(self):
        if not self.path.is_file():
            raise ValueError(f"Not an extant file: {self.path}")
        
    def as_string(self) -> str:
        return str(self.path)


@dataclass
class ExtantFolder:
    path: Path

    def __post_init__(self):
        if not self.path.is_dir():
            raise ValueError(f"Not an extant folder: {self.path}")
        
    def as_string(self) -> str:
        return str(self.path)
