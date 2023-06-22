"""Tools for working with paths"""

from dataclasses import dataclass
from pathlib import Path

__all__ = ["ExtantFile", "ExtantFolder"]
__author__ = "Vince Reuter"


class PathWrapperException(Exception):
    
    def __init__(self, original_exception):
        super(PathWrapperException, self).__init__(str(original_exception))
        self.original = original_exception


@dataclass
class PathWrapper:
    path: Path

    def __post_init__(self):
        if not isinstance(self.path, Path):
            raise ValueError(f"Not a path, but {type(self.path).__name__}: {self.path}")
    
    @classmethod
    def from_string(cls, rawpath: str):
        try:
            return cls(Path(rawpath))
        except Exception as e:
            raise PathWrapperException(e) from e

    def to_string(self) -> str:
        return str(self.path)


class ExtantFile(PathWrapper):

    def __post_init__(self):
        super(ExtantFile, self).__post_init__()
        if not self.path.is_file():
            raise ValueError(f"Not an extant file: {self.path}")
        

@dataclass
class ExtantFolder(PathWrapper):

    def __post_init__(self):
        super(ExtantFolder, self).__post_init__()
        if not self.path.is_dir():
            raise ValueError(f"Not an extant folder: {self.path}")
        

@dataclass
class NonExtantPath(PathWrapper):

    def __post_init__(self):
        super(NonExtantPath, self).__post_init__()
        if self.path.exists():
            raise ValueError(f"Path already exists: {self.path}")
