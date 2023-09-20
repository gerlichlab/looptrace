"""Apply quality control to traces and supporting points."""

import argparse
import os
from pathlib import Path
from typing import *

import yaml
from gertils import ExtantFile, ExtantFolder

from looptrace.ImageHandler import handler_from_cli
from looptrace.Tracer import Tracer
from looptrace.tracing_qc_support import apply_qc_filtration_and_write_results


def workflow(config_file: ExtantFile, images_folder: ExtantFolder) -> Tuple[Path, Path]:
    image_handler = handler_from_cli(config_file=config_file, images_folder=images_folder, image_save_path=None)
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    tracer = Tracer(image_handler=image_handler, array_id=None if array_id is None else int(array_id))
    traces_file = Path(tracer.traces_path)
    with open(config_file.path, 'r') as tmp_cfg:
        conf_data = yaml.safe_load(tmp_cfg)
    probes_to_ignore = conf_data["illegal_frames_for_trace_support"]
    min_trace_length = conf_data.get("min_trace_length", 0)
    print(f"Probes to ignore: {', '.join(probes_to_ignore)}")
    print(f"Min trace length: {min_trace_length}")
    return apply_qc_filtration_and_write_results(traces_file=traces_file, config_file=config_file.path, min_trace_length=min_trace_length, exclusions=probes_to_ignore)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply quality control to chromatin fiber traces and supporting points.')
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    parser.add_argument("image_path", type=ExtantFolder.from_string, help="Path to folder with images to read")
    args = parser.parse_args()
    workflow(config_file=args.config_path, images_folder=args.image_path)
