# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

import os
from pathlib import Path
from typing import *
import numpy as np


class Deconvolver:
    '''
    Class for handling generation and detection of e.g. nucleus images.
    '''

    def __init__(self, image_handler, array_id = None):
        self.image_handler = image_handler
        self.config = image_handler.config
        self.pos_list = self.image_handler.image_lists[self.config['decon_input_name']]

        if array_id is not None:
            self.pos_list = [self.pos_list[int(array_id)]]

    def extract_exp_psf(self) -> Path:
        '''
        Extract an experimental PDF from a bead image.
        Parameters read from config to segment same beads as used for drift correction.
        Segments and extract beads, filters the intensities to remove possible doublets,
        fits the centers, registers and overlays the beads, calculates an average and normalizes the signal.
        Saves as a .npy ndarray, image is a centered PSF useful for deconvolution.
        '''
        from scipy.ndimage import shift as shift_image
        from looptrace.gaussfit import fitSymmetricGaussian3DMLE
        from looptrace.image_processing_functions import extract_single_bead, generate_bead_rois

        t_slice = self.config['psf_bead_frame']
        ch = self.config['psf_bead_ch']
        threshold = self.config['bead_threshold']
        min_bead_int = self.config['min_bead_intensity']
        n_beads = 500
        bead_d = self.config.get('bead_roi_size', 16)
        bead_r = bead_d // 2
        
        bead_img = self.image_handler.images[self.config['psf_input_name']][0][t_slice, ch].compute()
        bead_pos = generate_bead_rois(bead_img, threshold, min_bead_int, bead_d, n_beads)
        beads = [extract_single_bead(point, bead_img) for point in bead_pos]
        bead_ints = np.sum(np.array(beads),axis = (1,2,3))
        perc_high = np.percentile(bead_ints, 40)
        perc_low = np.percentile(bead_ints, 5)
        beads = [b for b in beads if ((np.sum(b) < perc_high) & (np.sum(b) > perc_low))]

        fits = [fitSymmetricGaussian3DMLE(b, 3, [bead_r,bead_r,bead_r]) for b in beads]

        drifts = [np.array([bead_r, bead_r, bead_r]) - fit[0][2:5] for fit in fits]
        beads_c = [shift_image(b, d, mode='wrap') for b, d in zip(beads, drifts)]
        exp_psf = np.mean(np.stack(beads_c), axis=0)[1:, 1:, 1:]
        exp_psf = exp_psf / np.max(exp_psf)
        outfile = Path(self.image_handler.image_path) / "exp_psf.npy"
        print(f"Saving empirical point-spread function: {outfile}")
        np.save(str(outfile), exp_psf)
        print('Empirical PSF saved.')
        self.image_handler.images['exp_psf'] = exp_psf
        return outfile

    def decon_seq_images(self) -> List[str]:
        #Decovolve images using Flowdec.
        import dask.array as da
        import tqdm
        from looptrace.image_io import create_zarr_store
        from looptrace.image_processing_functions import decon_RL_setup

        decon_ch = self.config['decon_ch']
        if not isinstance(decon_ch, list):
            decon_ch = [decon_ch]
        non_decon_ch = self.config['non_decon_ch']
        if not isinstance(non_decon_ch, list):
            non_decon_ch = [non_decon_ch]
        n_iter = self.config['decon_iter']
        if n_iter == 0:
            print("Iterations set to 0.")
            return

        # TODO: better error handling for type-like errors, e.g. if comma is used for decimal 
        #       and as a result the value for a distance is parsed as string rather than number
        algo, psf, fd_data = decon_RL_setup(size_x=15, size_y=15, size_z=15, pz=0., wavelength=self.config['spot_wavelength']/1000,
            na=self.config['objective_na'], res_lateral=self.config['xy_nm']/1000, res_axial=self.config['z_nm']/1000)

        try: 
            psf_type = self.config['decon_psf']
        except:
            psf_type = 'gen'

        if psf_type == 'exp':
            try:
                psf =  self.image_handler.images['exp_psf']
                print('Using experimental psf for deconvolution.')
            except KeyError:
                print('Experimental PSF not extracted, extracting now.')
                self.extract_exp_psf()
                psf = self.image_handler.images['exp_psf']

        def run_decon(data, algo, fd_data, psf, n_iter):
            return algo.run(fd_data.Acquisition(data=data, kernel=psf), niter=n_iter).data.astype(np.uint16)
        decon_chunk = lambda chunk: run_decon(data=chunk, algo=algo, fd_data=fd_data, psf=psf, n_iter=n_iter)

        array_paths = []
        
        for pos in tqdm.tqdm(self.pos_list):
            pos_index = self.image_handler.image_lists[self.config['decon_input_name']].index(pos)
            pos_img = self.image_handler.images[self.config['decon_input_name']][pos_index]
            z = create_zarr_store(path=self.image_handler.image_save_path+os.sep+self.config['decon_input_name']+'_decon',
                    name = self.config['decon_input_name']+'_decon', 
                    pos_name = pos+'.zarr',
                    shape = (pos_img.shape[0], len(decon_ch) + len(non_decon_ch),) + pos_img.shape[-3:], 
                    dtype = np.uint16, 
                    chunks = (1, 1, 1, pos_img.shape[-2], pos_img.shape[-1]))

            for i, t_img_full in tqdm.tqdm(enumerate(pos_img)):
                for ch in decon_ch:
                    t_img = np.array(t_img_full[ch])
                    if np.any(np.array(t_img.shape[-3:])<5):
                        t_img = np.zeros_like(t_img)
                    elif np.any(np.array(t_img.shape) > 1000):
                        if t_img.shape[0] > 100:
                            chunk_size = (100, 900, 900)
                            depth = (4,8,8)
                        else:
                            chunk_size = (t_img.shape[0], 900, 900)
                            depth = (0,8,8)

                        arr = da.from_array(t_img, chunks=chunk_size)
                        t_img = arr.map_overlap(decon_chunk,depth=depth, boundary='reflect', dtype='uint16').compute(num_workers=1)
                    else:
                        t_img = run_decon(data=t_img, algo=algo, fd_data=fd_data, psf=psf, n_iter=n_iter)
                    print(t_img.shape)
                    z[i, ch] = t_img.copy()

                for ch in non_decon_ch:
                    z[i, ch] = np.array(t_img_full[ch])
            
            array_paths.append(z.store.path)

        return array_paths
