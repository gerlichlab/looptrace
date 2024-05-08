"""Partition full-experiment regional spots file(s) by field of view (FOV)."""

import argparse
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
        "-N",
        "--nuclei-filtered-spots-file",
        type=Path,
        help="Path to nuclei-filtered regional spots file",
    )
    parser.add_argument(
        "-P",
        "--proximity-filtered-spots-file",
        type=Path,
        help="Path to proximity-filtered regional spots file",
    )
    parser.add_argument(
        "-U",
        "--unfiltered-spots-file",
        type=Path,
        help="Path to unfiltered regional spots file",
    )
    parser.add_argument(
        "-O",
        "--output-folder",
        required=True,
        type=Path,
        help="Folder in which to write output",
    )
    return parser.parse_args(cmdl)


def workflow(*, output_folder: Path, spots_files: list[Path]) -> dict[str, list[Path]]:
    """Main workhorse of this script"""
    outputs: dict[str, list[Path]] = {}
    for fp in spots_files:
        logging.info("Reading spots file: %s", fp)
        spots_table = pd.read_csv(fp, index_col=0)
        for position, subtable in spots_table.groupby("position"):
            position = position.rstrip(".zarr")
            target = output_folder / position / f"{position}__{fp.name}"
            target.parent.mkdir(exist_ok=True, parents=True)
            logging.info("Writing data for FOV %s: %s", position, target)
            subtable.reset_index(drop=True).to_csv(target)
            outputs.setdefault(position, []).append(target)
    return outputs


def main(cmdl: list[str]) -> None:
    """Driver function of this script"""
    opts = parse_cmdl(cmdl)
    spots_files: list[Path] = [
        f
        for f in [
            opts.nuclei_filtered_spots_file,
            opts.proximity_filtered_spots_file,
            opts.unfiltered_spots_file,
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
