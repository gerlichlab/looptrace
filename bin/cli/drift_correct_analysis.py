"""Quality control / analysis of the drift correction step"""

import argparse
import sys
from pathlib import Path
from typing import *

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import seaborn as sns
import tqdm

from gertils.pathtools import ExtantFile, ExtantFolder
from looptrace.gaussfit import fitSymmetricGaussian3D
from looptrace import image_io
from looptrace import image_processing_functions as ip


def parse_cmdl(cmdl: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quality control / analysis of the drift correction step", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("--output-folder", required=True, type=Path, help="Path to output folder")
    parser.add_argument("--images-folder", required=True, type=ExtantFolder.from_string, help="Path to folder with images used for drift correction")
    parser.add_argument("--drift-correction-table", required=True, type=ExtantFile.from_string, help="Path to drift correction table")
    parser.add_argument("--reference-FOV", type=int, help="0-based index, referring to fields of view, for using as reference.")
    return parser.parse_args(cmdl)


def process_single_FOV_single_reference_frame(imgs: List[np.ndarray], drift_table: pd.DataFrame, full_pos: int, ref_frame: int, ref_ch: int) -> pd.DataFrame:
    T = imgs[full_pos].shape[0]
    C = imgs[full_pos].shape[1]
    ref_rois = ip.generate_bead_rois(imgs[full_pos][ref_frame,ref_ch].compute(), 2000, 5000, n_points = -1)
    rois = ref_rois[np.random.choice(ref_rois.shape[0], 50, replace=False)]
    bead_roi_px = 8
    dims = (len(rois),T,C,bead_roi_px,bead_roi_px,bead_roi_px)
    bead_imgs = np.zeros(dims)
    bead_imgs_dc = np.zeros(dims)

    fits = []
    for t in tqdm.tqdm(range(T)):
        # TODO: this requires that the drift table be ordered such that the FOVs are as expected; need flexibility.
        pos = drift_table.position.unique()[full_pos]
        course_shift = drift_table[(drift_table.position == pos) & (drift_table.frame == t)][['z_px_course', 'y_px_course', 'x_px_course']].values[0]
        fine_shift = drift_table[(drift_table.position == pos) & (drift_table.frame == t)][['z_px_fine', 'y_px_fine', 'x_px_fine']].values[0]
        for c in [ref_ch]:#range(C):
            img = imgs[full_pos][t, c].compute()
            for i, roi in enumerate(rois):
                bead_img=ip.extract_single_bead(roi, img, bead_roi_px=bead_roi_px, drift_course=course_shift)
                fit = fitSymmetricGaussian3D(bead_img, sigma=1, center='max')[0]
                fits.append([full_pos, t, c, i] + list(fit))
                bead_imgs[i,t,c] = bead_img.copy()
                bead_imgs_dc[i,t,c] = ndi.shift(bead_img, shift=fine_shift)

    fits = pd.DataFrame(fits, columns=['full_pos','t', 'c', 'roi', 'BG', 'A', 'z_loc', 'y_loc', 'x_loc', 'sigma_z', 'sigma_xy'])
    
    # TODO: parameterise these factors with the config and/or CLI
    xy_scale_factor = 110
    z_scale_factor = 300
    fits.loc[:, ['y_loc', 'x_loc', 'sigma_xy']] = fits.loc[:,  ['y_loc', 'x_loc', 'sigma_xy']] * xy_scale_factor #Scale xy coordinates to nm (use xy pixel size from exp)
    fits.loc[:, ['z_loc', 'sigma_z']] = fits.loc[:, ['z_loc', 'sigma_z']] * z_scale_factor #Scale z coordinates to nm (use slice spacing from exp)

    ref_points = fits.loc[(fits.t == ref_frame) & (fits.c == 0), ['z_loc', 'y_loc', 'x_loc']].to_numpy() #Fits of fiducial beads in ref frame
    res = []
    for t in tqdm.tqdm(range(T)):
        mov_points = fits.loc[(fits.t == t) & (fits.c == 0), ['z_loc', 'y_loc', 'x_loc']].to_numpy() #Fits of fiducial beads in moving frame
        shift = drift_table.loc[(drift_table.position == pos) & (drift_table.frame == t), ['z_px_fine', 'y_px_fine', 'x_px_fine']].values[0]
        shift[0] =  shift[0] * z_scale_factor #Extract calculated drift correction from drift correction file.
        shift[1] =  shift[1] * xy_scale_factor
        shift[2] =  shift[2] * xy_scale_factor
        fits.loc[(fits.t == t), ['z_dc', 'y_dc', 'x_dc']] = mov_points + shift #Apply precalculated drift correction to moving fits
        fits.loc[(fits.t == t), ['z_dc_rel', 'y_dc_rel', 'x_dc_rel']] =  np.abs(fits.loc[(fits.t == t), ['z_dc', 'y_dc', 'x_dc']].to_numpy() - ref_points)#Find offset between moving and reference points.
        fits.loc[(fits.t == t), ['euc_dc_rel']] = np.sqrt(np.sum((fits.loc[(fits.t == t), ['z_dc', 'y_dc', 'x_dc']].to_numpy() - ref_points)**2, axis=1)) #Calculate 3D eucledian distance between points and reference
        res.append(shift)

    # TODO: paramterise by config and/or CLI.
    fits['A_to_BG'] = fits['A'] / fits['BG']
    fits['QC'] = 0
    fits.loc[fits['A_to_BG'] > 2, 'QC'] = 1

    return fits


def workflow(images_folder: Path, drift_correction_table_file: Path, output_folder: Path, full_pos: Optional[int] = None) -> pd.DataFrame:
    # TODO: how to handle case when output already exists
    # TODO: how to iterate over or aggregate the FOVs as reference

    print(f"Reading zarr to dask: {images_folder}")
    imgs, _ = image_io.multi_ome_zarr_to_dask(images_folder)
    print(f"Reading drift correction table: {drift_correction_table_file}")
    drift_table = pd.read_csv(drift_correction_table_file, index_col=0)

    proc_1_fov = lambda pos_idx: process_single_FOV_single_reference_frame(
        imgs=imgs, 
        drift_table=drift_table, 
        full_pos=pos_idx,
        ref_frame=10, # TODO: parameterise with config.
        ref_ch=0, # TODO: parameterise with config.
        )
    
    if full_pos is not None:
        fits = proc_1_fov(full_pos)
        make_plot = True
    else:
        fits = pd.concat(map(proc_1_fov, range(len(drift_table.position.unique()))))
        make_plot = False
    
    fits_output_file = output_folder / "dc_analysis_fits.tsv"
    print(f"Writing fits file: {fits_output_file}")
    fits.to_csv(fits_output_file, index=False, sep="\t")

    if make_plot:
        plot_fits(fits)

    return fits
    

def plot_fits(fits: pd.DataFrame, outfile: Path):
    print("Plotting...")
    sns.lineplot(data = fits[(fits.QC==1) & (fits.c ==0)], y = 'z_dc_rel', x='t',  estimator=np.median)
    sns.lineplot(data = fits[(fits.QC==1) & (fits.c ==0)], y = 'y_dc_rel', x='t',  estimator=np.median)
    sns.lineplot(data = fits[(fits.QC==1) & (fits.c ==0)], y = 'x_dc_rel', x='t',  estimator=np.median)
    sns.lineplot(data = fits[(fits.QC==1) & (fits.c ==0)], y = 'euc_dc_rel', x='t',  estimator=np.median)
    print(fits[(fits.QC==1) & (fits.c ==0)].z_dc_rel.median(), fits[(fits.QC==1) & (fits.c ==0)].y_dc_rel.median(), fits[(fits.QC==1) & (fits.c ==0)].x_dc_rel.median(), fits[(fits.QC==1) & (fits.c ==0)].euc_dc_rel.median())
    print(fits[(fits.QC==1) & (fits.c ==0)].z_dc_rel.quantile([0.25,0.75]), fits[(fits.QC==1) & (fits.c ==0)].y_dc_rel.quantile([0.25,0.75]), fits[(fits.QC==1) & (fits.c ==0)].x_dc_rel.quantile([0.25,0.75]), fits[(fits.QC==1) & (fits.c ==0)].euc_dc_rel.quantile([0.25,0.75]))
    print(fits[(fits.QC==1) & (fits.c ==0)].z_dc_rel.mean(), fits[(fits.QC==1) & (fits.c ==0)].y_dc_rel.mean(), fits[(fits.QC==1) & (fits.c ==0)].x_dc_rel.mean(), fits[(fits.QC==1) & (fits.c ==0)].euc_dc_rel.mean())
    print(fits[(fits.QC==1) & (fits.c ==0)].z_dc_rel.std(), fits[(fits.QC==1) & (fits.c ==0)].y_dc_rel.std(), fits[(fits.QC==1) & (fits.c ==0)].x_dc_rel.std(), fits[(fits.QC==1) & (fits.c ==0)].euc_dc_rel.std())
    plt.ylim(0,150)
    print(f"Saving plot: {outfile}")
    plt.savefig(outfile, dpi=300)
    

if __name__ == "__main__":
    # TODO: setup logger with logmuse
    opts = parse_cmdl(sys.argv[1:])
    workflow(
        images_folder=opts.images_folder.path, 
        drift_correction_table_file=opts.drift_correction_table.path, 
        output_folder=opts.output_folder, 
        full_pos=opts.reference_FOV
    )
