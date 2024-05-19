"""Organising the data on disk for drag-and-drop visualisation with the Napari plugin looptrace-loci-vis"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional

from gertils.types import FieldOfViewFrom1
from looptrace.integer_naming import get_position_name_short


logger = logging.getLogger(__name__)

CommandChunks = list[str]


def parse_cmdl(cmdl: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organising the data on disk for drag-and-drop visualisation with the Napari plugin looptrace-loci-vis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-I", "--input-folder", required=True, type=Path, help="Path to folder in which to find files")
    parser.add_argument("-O", "--output-folder", type=Path, help="Path to folder in which files should be placed; use input folder if unspecified")
    parser.add_argument("--script", required=True, type=Path, help="Path to script to which to write commands")
    return parser.parse_args(cmdl)


def get_fov_inference_substrate(fp: Path) -> str:
    return fp.name.split(".")[0]


def get_fov(s: str) -> Optional[FieldOfViewFrom1]:
    prefix = "P"
    if not s.startswith(prefix):
        return None
    try:
        raw_val = int(s.lstrip(prefix))
    except (TypeError, ValueError):
        return None
    return FieldOfViewFrom1(raw_val)


def find_files(infolder: Path) -> list[tuple[FieldOfViewFrom1, Path]]:
    result: list[tuple[FieldOfViewFrom1, Path]] = []
    for fp in infolder.iterdir():
        fov: Optional[FieldOfViewFrom1] = None
        fov_inference_substrate: str = get_fov_inference_substrate(fp)
        if fp.is_file() and fp.name.endswith(".qcpass.csv") or fp.name.endswith(".qcfail.csv"):
            logger.debug("Inferring FOV for file: %s", fp)
            fov = get_fov(fov_inference_substrate)
        elif fp.is_dir() and fp.suffix == ".zarr":
            logger.debug("Inferring FOV for folder: %s", fp)
            fov = get_fov(fov_inference_substrate)
        if fov is not None:
            result.append((fov, fp))
        else:
            logger.debug("No FOV inferred, ignoring path: %s", fp)
    return result


def workflow(*, infolder: Path, outfolder: Optional[Path] = None) -> list[tuple[Path, Path]]:
    if outfolder is None:
        logger.debug("Using input folder for output: %s", infolder)
        outfolder = infolder
    files = find_files(infolder)
    by_fov: dict[FieldOfViewFrom1, list[Path]] = {}
    for fov, fp in files:
        by_fov.setdefault(fov, []).append(fp)
    src_dst_pairs: list[tuple[Path, Path]] = []
    for fov, files_group in by_fov.items():
        subfolder = outfolder / get_position_name_short(fov.get - 1)
        if not subfolder.is_dir():
            logger.info("Creating folder: %s", subfolder)
            subfolder.mkdir(parents=True)
        for fp in files_group:
            src_dst_pairs.append((fp, subfolder / fp.name))
    return src_dst_pairs


def main(cmdl: list[str]) -> None:
    opts = parse_cmdl(cmdl)
    src_dst_pairs: list[tuple[Path, Path]] = workflow(infolder=opts.input_folder, outfolder=opts.output_folder)
    commands = [f"mv {src} {dst}" for src, dst in src_dst_pairs]
    logger.info("Command count: %d", len(commands))
    logger.info("Writing data movement/organisation script: %s", opts.script)
    with opts.script.open(mode="w") as fh:
        for cmd in commands:
            fh.write(cmd + "\n")
    logger.info("Done!")


if __name__ == "__main__":
    main(sys.argv[1:])
