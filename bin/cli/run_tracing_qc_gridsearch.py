"""Count chromatin trace supports passing QC over a range of parameter settings."""

import argparse
import sys
from pathlib import Path
from typing import *

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from gertils import ExtantFile, ExtantFolder

from tracing_qc import QC_PASS_COUNT_COLUMN, write_tracing_qc_passes_from_gridfile

__author__ = "Vince Reuter"


def parse_cmdl(cmdl: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count chromatin trace supports passing QC over a range of parameter settings.", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-T", "--tracesFile", type=ExtantFile.from_string, required=True, help="Path to traces table file, from end of looptrace")
    parser.add_argument("-C", "--configFile", type=ExtantFile.from_string, required=True, help="Path to main looptrace config file")
    parser.add_argument("-G", "--gridFile", type=ExtantFile.from_string, required=True, help="Path to QC parameters grid file")
    parser.add_argument("-O", "--outputFolder", type=ExtantFolder.from_string, required=True, help="Path to folder in which to place ouput")
    parser.add_argument("-X", "--probeExclusions", nargs='*', help="Names of probes/frames to exclude")
    parser.add_argument("--qcCountVariables", nargs='*', help="Variables to plot effect for QC pass counts")
    return parser.parse_args(cmdl)


def main(cmdl: List[str]):
    opts = parse_cmdl(cmdl)
    
    def get_output_file(fn: str) -> Path:
        return opts.outputFolder.path / fn

    def save_scatterplot(fn: str, data: pd.DataFrame, x: str, y: str, **kwargs) -> Path:
        plotfile = get_output_file(fn)
        sns.scatterplot(data=data, x=x, y=y)
        if "xlim" in kwargs:
            plt.xlim(*kwargs["xlim"])
        if "ylim" in kwargs:
            plt.ylim(*kwargs["ylim"])
        print(f"Saving plot: {plotfile}")
        plt.savefig(plotfile)
        plt.close()
        return plotfile

    trace_table, qc_pass_counts = write_tracing_qc_passes_from_gridfile(
        traces_file=opts.tracesFile, 
        config_file=opts.configFile, 
        gridfile=opts.gridFile, 
        outfile=get_output_file("qc_pass_counts.csv"), 
        exclusions=opts.probeExclusions,
        )
    
    print(trace_table.shape)
    print(qc_pass_counts.shape)
    
    trace_table["SNR"] = trace_table["A"] / trace_table["BG"]
    correlations = trace_table[["sigma_xy", "sigma_z", "SNR"]].corr()
    print(f"Correlations:\n{correlations}")
    corr_out_file = get_output_file("correlations.csv")
    print(f"Saving correlations: {corr_out_file}")
    correlations.to_csv(corr_out_file, index=False)
    print("Saving correlation-related scatterplots...")
    save_scatterplot("SNR__sd_xy.png", data=trace_table, x="sigma_xy", y="SNR", xlim=(0, 1000), ylim=(0, 100))
    save_scatterplot("SNR__sd_z.png", data=trace_table, x="sigma_z", y="SNR", xlim=(0, 1000), ylim=(0, 100))
    
    for xvar in opts.qcCountVariables:
        print(f"Plotting effect of '{xvar}' on QC pass count...")
        save_scatterplot(f"QC_pass__{xvar}.png", data=qc_pass_counts, x=xvar, y=QC_PASS_COUNT_COLUMN)


if __name__ == "__main__":
    main(sys.argv[1:])
