"""Partition full-experiment regional spots file(s) by field of view (FOV)."""

import argparse
from collections.abc import Iterable
import logging
from pathlib import Path
import sys

import pandas as pd


def parse_cmdl(cmdl: list[str]) -> argparse.Namespace:
    """Define and parse the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Partition full-experiment regional spots file(s) by field of view (FOV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--merge-contributors-file",
        type=Path,
        help="Path to regional spots file of ROIs which contributed to a merger",
    )
    parser.add_argument(
        "--proximity-discards-file",
        type=Path,
        help="Path to regional spots file of ROIs discarded due to being too close together",
    )
    parser.add_argument(
        "--nuclei-labeled-file",
        type=Path,
        help="Path to regional spots file of ROIs which passed proximity filtration and are labeled with nuclus assignment",
    )
    parser.add_argument(
        "-O",
        "--output-folder",
        required=True,
        type=Path,
        help="Folder in which to write output",
    )
    return parser.parse_args(cmdl)


def workflow(*, output_folder: Path, spots_files: Iterable[Path]) -> dict[str, list[Path]]:
    """Main workhorse of this script"""
    outputs: dict[str, list[Path]] = {}
    for fp in spots_files:
        logging.info("Reading spots file: %s", fp)
        spots_table = pd.read_csv(fp, index_col=0)
        for fov, subtable in spots_table.groupby("fieldOfView"):
            fov = fov.rstrip(".zarr")
            target = output_folder / fov / f"{fov}__{fp.name}"
            target.parent.mkdir(exist_ok=True, parents=True)
            logging.info("Writing data for FOV %s: %s", fov, target)
            subtable.reset_index(drop=True).to_csv(target)
            outputs.setdefault(fov, []).append(target)
    return outputs


def main(cmdl: list[str]) -> None:
    """Driver function of this script"""
    opts = parse_cmdl(cmdl)
    spots_files: list[Path] = [
        f
        for f in [
            opts.merge_contributors_file,
            opts.proximity_discards_file,
            opts.nuclei_labeled_file,
        ]
        if f is not None
    ]
    if len(spots_files) == 0:
        raise RuntimeError("Nothing to filter!")
    logging.info("%d spots file(s): %s", len(spots_files), ", ".join(map(str, spots_files)))
    outputs: dict[str, list[Path]] = workflow(
        output_folder=opts.output_folder, spots_files=spots_files
    )
    logging.info("Wrote data for %d FOVs", len(outputs))
    logging.info("Output file count: %d", sum(len(fs) for fs in outputs.values()))
    logging.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
