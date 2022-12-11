"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from looptrace.ImageHandler import ImageHandler
import pandas as pd
import os
import argparse
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean up output files from cluster processing.')
    parser.add_argument("config_path", help="Config file path")
    args = parser.parse_args()
    H = ImageHandler(config_path=args.config_path)

    all_files = os.scandir(H.config['analysis_path'])
    sel_files = sorted([f.path for f in all_files if re.match(".*\d{4}.csv", f.name)])
    if len(sel_files) > 0:
        out_path = H.out_path+sel_files[0][len(H.out_path):-9]+'.csv'

        dfs = [pd.read_csv(f, index_col=0) for f in sel_files]
        dfs = pd.concat(dfs).sort_values(['position','frame']).reset_index(drop=True)
        dfs.to_csv(out_path)
        [os.remove(f) for f in sel_files]
        print('Files combined.')
    else:
        print('No files to clean up.')