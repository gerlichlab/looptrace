"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import argparse
import os
import re
import pandas as pd

from looptrace.ImageHandler import handler_from_cli
from gertils import ExtantFile


def workflow(config_file: ExtantFile) -> None:
    image_handler = handler_from_cli(config_file=config_file, images_folder=None, image_save_path=None)
    all_files = os.scandir(image_handler.config['analysis_path'])
    sel_files = sorted([f.path for f in all_files if re.match(".*\d{4}.csv", f.name)])
    if len(sel_files) > 0:
        # TODO: better solution
        dummy_index = len(os.path.join(image_handler.config['analysis_path'], image_handler.config['analysis_prefix']).rstrip(os.pathsep))
        #out_path = image_handler.out_path(sel_files[0][len(image_handler.out_path):-9] + '.csv')
        out_path = image_handler.out_path(sel_files[0][dummy_index:-9] + '.csv')

        dfs = [pd.read_csv(f, index_col=0) for f in sel_files]
        dfs = pd.concat(dfs).sort_values(['position','frame']).reset_index(drop=True)
        print(f"Writing combined data: {out_path}")
        dfs.to_csv(out_path)
        print('Files combined.')
        print("Cleaning up...")
        for f in sel_files:
            os.remove(f)
        print("Cleanup complete.")
    else:
        print('No files to clean up.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean up output files from cluster processing.')
    parser.add_argument("config_path", type=ExtantFile.from_string, help="Config file path")
    args = parser.parse_args()
    workflow(config_file=args.config_path)
