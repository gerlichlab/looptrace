"""This program 'squeezes' a list of ZARR files into another list, renumbering FOVs so that order is preserved but sequence is [1, ..., N]."""

import argparse
import logging
from operator import itemgetter
from pathlib import Path
import sys

from gertils.types import FieldOfViewFrom1

from looptrace.integer_naming import NameableSemantic, get_position_name_short


ZARR_EXTENSION = ".zarr"


def parse_cmdl(cmdl: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="This program 'squeezes' a list of ZARR files into another list, renumbering FOVs so that order is preserved but sequence is [1, ..., N].",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data_folder", type=Path, help="Path to folder in which to find files")
    parser.add_argument("--script-file", required=True, type=Path, help="Path to file to write as script")
    parser.add_argument("--overwrite-script", action="store_true", help="Say that overwriting script should be allowed")
    return parser.parse_args(cmdl)


def workflow(*, data_folder: Path, script_file: Path) -> None:
    path_by_fov: dict[FieldOfViewFrom1, Path] = {}
    for f in data_folder.iterdir():
        if not (f.is_dir() and f.name.startswith(NameableSemantic.Point.value) and f.suffix == ZARR_EXTENSION):
            logging.debug("Not considering path: %s", f)
        try:
            raw_fov = int(f.stem.lstrip(NameableSemantic.Point.value))
        except (TypeError, ValueError) as e:
            logging.debug("Cannot parse FOV from file %s: %s", f, e)
            continue
        fov = FieldOfViewFrom1(raw_fov)
        if fov in path_by_fov:
            raise KeyError(f"Repeat FOV ({fov}) in folder {data_folder}")
        path_by_fov[fov] = f
    logging.info("Writing script: %s", script_file)
    with script_file.open(mode="w") as fh:
        for new_raw_fov_zero_based, (_, old_path) in enumerate(sorted(path_by_fov.items(), key=itemgetter(0))):
            new_name = get_position_name_short(new_raw_fov_zero_based) + ZARR_EXTENSION
            new_path = old_path.parent / new_name
            cmd = f"mv {old_path} {new_path}"
            fh.write(cmd + "\n")
    logging.info("Done!")


def main(cmdl: list[str]) -> None:
    opts = parse_cmdl(cmdl)
    logging.basicConfig(level=logging.INFO)
    if opts.script_file.exists() and not opts.overwrite_script:
        raise FileExistsError(f"Script to write already exists: {opts.script_file}")
    workflow(data_folder=opts.data_folder, script_file=opts.script_file)


if __name__ == "__main__":
    main(sys.argv[1:])
