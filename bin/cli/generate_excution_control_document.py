"""Script to generate the pipeline execution control documentation"""

import argparse
import os
import sys
from typing import *

__author__ = "Vince Reuter"

from run_processing_pipeline import DECON_STAGE_NAME, PIPE_NAME, SPOT_DETECTION_STAGE_NAME, TRACING_QC_STAGE_NAME, LooptracePipeline


def _parse_cmdl(cmdl: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the pipeline exceution control documentation.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument("-O", "--outfile", required=True, help="Path to output file")
    parser.add_argument(
        "--stage-names", 
        nargs='*', 
        default=[DECON_STAGE_NAME, SPOT_DETECTION_STAGE_NAME, TRACING_QC_STAGE_NAME], 
        help="Names of stages for which specific predecessor steps should be noted",
        )
    return parser.parse_args(cmdl)


DOCTEXT = f"""<!--- DO NOT EDIT THIS GENERATED DOCUMENT DIRECTLY; instead, edit {os.path.basename(__file__)} --->
# Controlling pipeline execution
## Overview
The main `looptrace` processing pipeline is built with [pypiper](https://pypi.org/project/piper/).
One feature of such a pipeline is that it may be started and stopped at arbitrary points.
To do so, the start and end points must be specified by name of processing stage.

To __start__ the pipeline from a specific point, use `--start-point <stage name>`. Example:
```python
python run_processing_pipeline.py -C conf.yaml -I images_folder -O pypiper_output --start-point {SPOT_DETECTION_STAGE_NAME} --stop-before {TRACING_QC_STAGE_NAME}
```

To __stop__ the pipeline...<br>
* ...just _before_ a specific point, use `--stop-before <stage name>`.
* ...just _after_ a specific point, use `--stop-after <stage name>`.

## Restarting the pipeline...
When experimenting with different parameter settings for one or more stages, it's common to want to restart the pipeline from a specific point.
Before rerunning the pipeline with the appropriate `--start-point` value, take care of the following:

1. __Analysis folder__: It's wise to create a new analysis / output folder for a restart, particularly if it corresponds to updated parameter settings.
1. __Configuration file__: It's wise to create a new config file for a resart if it corresponds to updated parameter settings. 
Regardless of whether that's done, ensure that the `analysis_path` value corresponds to the output folder you'd like to use.
1. __Pipeline (pypiper) folder__: You should create a new pypiper folder for a restart with new parameters.
This is critical since the semaphore / checkpoint files will influence pipeline control flow.
You should copy to this folder any checkpoint files of any stages upstream of the one from which you want the restart to begin.
Even though `--start-point` should allow the restart to begin from where's desired, if that's forgotten the checkpoint files should save you.

Generate an empty checkpoint file for each you'd like to skip. 
Simply create (`touch`) each such file `{PIPE_NAME}_<stage>.checkpoint` in the desired pypiper output folder.
Below are the sequential pipeline stage names.

### Pipeline stage names
"""


def get_stage_predecessors_text(stage: str, preds: List[str]) -> List[str]:
    return [f"### ...`{stage}`"] + [f"* {name}" for name in preds]


def main(cmdl):
    opts = _parse_cmdl(cmdl)
    stage_names = [n for n, _, _ in LooptracePipeline.name_fun_getargs_bundles()]
    full_text = DOCTEXT + "\n".join(f"* {sn}" for sn in stage_names)
    print(f"Writing docs file: {opts.outfile}")
    with open(opts.outfile, 'w') as out:
        out.write(full_text)
    print("Done!")


if __name__ == "__main__":
    main(sys.argv[1:])
