# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg

Usage:
    - Update YAML config file.
    - from looptrace import Compare
    - Initialize comparison object: C = Compare(path_to_yaml_file)
    - run C.compare_multi_sc()
"""

import os

import dask.array as da
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.exposure import match_histograms
from skimage.measure import regionprops_table
from skimage.filters import threshold_otsu
import tifffile as tiff
import yaml

from looptrace import comparison_functions as comp
from looptrace import image_processing_functions as ip
from looptrace.wrappers import phase_xcor


class Compare:
    def __init__(self, config_path):
        '''
        Initialize Image Processing class with config read in from YAML file.
        '''
        
        with open(config_path, 'r') as fh:
            self.config = yaml.load(fh)
        self.config_path = config_path
        self.zarr_path = self.config['output_folder']+os.sep+self.config['output_name']+'.zarr'
        if os.path.isdir(self.zarr_path):
            self.images = da.from_zarr(self.zarr_path)
        else:
            self.images = self.load_comp_images()
            self.save_zarr()
            self.images = da.from_zarr(self.zarr_path)
        
    def reload_config(self):
        with open(self.config_path, 'r') as fh:
            self.config = yaml.load(fh)
    
    def load_comp_images(self):
        config=self.config
        print(config)
        template1=config['template_comp_1']+config['filetype']
        template2=config['template_comp_2']+config['filetype']
        input1=config['comp_folder_1']
        input2=config['comp_folder_2']
        
        images1, _ = ip.images_to_dask(input1,template1)
        images2, _ = ip.images_to_dask(input2,template2)
        images = da.concatenate([images1, images2], axis=1)
        return images

    def drift_correction(self, ds: int = 2) -> np.ndarray:
        """
        Global 3D drift correction of images to compare

        Parameters
        ----------
        ds : int, optional
            Downsampling for drift correction, defaulting to 2

        Returns
        -------
        list of np.ndarray
            List of global drifts per position found for comparison.
        """

        dc_ch = self.config['drift_correction_channel']
        drifts = []
        for pos in range(self.images.shape[0]):
            print(f'Drift correcting position {pos}.')
            img1 = self.images[pos, 0, dc_ch, ::ds, ::ds, ::ds].compute()
            print('Loaded image 1')
            img2 = self.images[pos, 1, dc_ch, ::ds, ::ds, ::ds].compute()
            print('Loaded image 2')
            drift = np.array(phase_xcor(img1, img2)) * ds
            print(f'Found drift {drift}.')
            drifts.append(drift)
        self.drifts = drifts
        return drifts

    def detect_nuclei(self, ds=2):
        '''
        Runs nucleus detection from ip for comparison of single nuclei.
        Makes 2D slices at the central plane of the nucleus for reliable comparison.
        Drift corrects again to ensure best possible overlap.
        Ca

        Args:
            ds (int, optional): Downsampling for detection and comparison. Defaults to 2.

        Returns:
            nucs [DataFrame]: Bounding boxes for detected nuclei
            z_img_nucs [np array]: Center slices of detected nuclei. 
        '''
        import imreg_dft

        nuc_ch = self.config['nuc_ch']          #Channel of nucleus.
        nuc_d = self.config['nuc_diameter']/ds  #Approximate diameter of nucleus in pixels.
        drifts = self.drifts                    #Precalculated table of 3d image drifts
        z_o = self.config['z_offset']//ds       #Offset from central z-slice to use (e.g. for top of nucleus)
        nucs = []
        z_nuc_imgs = []
        for pos in range(self.images.shape[0]):
            print('Analysing pos ', pos)
            #Read drifts, scale by downsampling.
            d_z = drifts[pos][0]//ds
            d_y = drifts[pos][1]//ds
            d_x = drifts[pos][2]//ds

            #Read nuclear images.
            img_t = self.images[pos, 0, nuc_ch, ::ds, ::ds, ::ds].compute()
            img_o = self.images[pos, 1, nuc_ch, ::ds, ::ds, ::ds]

            #Detect nuclear masks and dilate them a bit before calculating bounding boxes.
            masks = ip.nuc_segmentation(np.max(img_t, axis=0), nuc_d)
            labels, n_nucs = ndi.label(masks)
            labels = ndi.morphology.grey_dilation(labels, 20//ds)
            bbox = pd.DataFrame(regionprops_table(labels, properties=('label',
                                                                    'bbox',
                                                                    'area')))
            bbox['position'] = pos
            bbox['ds'] = ds
            
            #Exclude very small nuclei.
            bbox = bbox[bbox['area'] > nuc_d**2/10]
            for i, row in bbox.iterrows():
                print('Analysing nuc', i)
                #Select single nuclei from larger image.
                ymin = row['bbox-0']
                xmin = row['bbox-1']
                ymax = row['bbox-2']
                xmax = row['bbox-3']

                nuc_img_t = img_t[:,ymin:ymax, xmin:xmax]

                #Find central plane (plane with max intensity in nucleus)
                zmid = min(np.argmax(np.sum(nuc_img_t, axis=(1,2))) + z_o, nuc_img_t.shape[0]-1)

                bbox.loc[i, 'zmid'] = zmid
                nuc_img_t = nuc_img_t[zmid]

                #Iterate over potential matching planes in offset image, find closest match
                zmin = int(np.clip(zmid-d_z-6//ds, 0, img_o.shape[0]-1))
                zmax = int(np.clip(zmid-d_z+6//ds, 0, img_o.shape[0]-1))
                ymin = np.clip(ymin-d_y, 0, img_o.shape[1]-1)
                ymax = np.clip(ymax-d_y, 0, img_o.shape[1]-1)
                xmin = np.clip(xmin-d_x, 0, img_o.shape[2]-1)
                xmax = np.clip(xmax-d_x, 0, img_o.shape[2]-1)
                
                z_scan_range = list(range(zmin, zmax))
                nuc_imgs_o_temp = []
                corrs = []
                try:
                    for z in z_scan_range:
                        nuc_img_o_temp = img_o[z, ymin:ymax, xmin:xmax].compute()
                    
                        nuc_img_o_temp = ip.pad_to_shape(nuc_img_o_temp, nuc_img_t.shape)
                        nuc_img_o_temp = imreg_dft.similarity(nuc_img_t, nuc_img_o_temp, numiter=10, constraints=
                                                        {'scale':[1,0],
                                                            'angle':[0,10],
                                                            'tx':[0,5],
                                                            'ty':[0,5]})['timg']
                        
                        nuc_imgs_o_temp.append(nuc_img_o_temp)
                        corrs.append(comp.comp_pcc_coloc(nuc_img_t, nuc_img_o_temp))   
                    max_corr_index = np.argmax(np.array(corrs))    
                except (ValueError, IndexError): #In case drift correction has gone horribly wrong.
                            continue

                zmid_o = z_scan_range[max_corr_index]
                nuc_img_o = nuc_imgs_o_temp[max_corr_index]
                print('Z-stack correlations are:', corrs)
                bbox.loc[i, 'zmid_o'] = zmid_o

                #Calculate residual fine-scale drift:
                new_drift = phase_xcor(nuc_img_t, nuc_img_o, upsample_factor=2)
                bbox.loc[i, 'y_f'] = new_drift[0]
                bbox.loc[i, 'x_f'] = new_drift[1]
                nuc_img_o = ndi.shift(nuc_img_o, new_drift)
                nuc_img_o = match_histograms(nuc_img_o, nuc_img_t)
                z_nuc_imgs.append(np.stack([nuc_img_t, nuc_img_o]))
            nucs.append(bbox)
            print(f'Detected {n_nucs} nuclei.')
        nucs = pd.concat(nucs, axis=0).reset_index(drop=True)
        self.nucs = nucs
        self.z_nuc_imgs = z_nuc_imgs
        return nucs, z_nuc_imgs

    def detect_rds(self, ds=2):
        '''
        Detect and segment replication domains (or other features) by otsu threshold.
        Makes maximum projections for more consistent comparison.
        Align comparison images of features and recalculate fine scale drift.

        Args:
            ds (int, optional): Downscaling factor for comparison feature images. Defaults to 2.

        Returns:
            rds [DataFrame]: Bounding boxes of all the detected features.
            rd_imgs [np array]: Max projection images of detected features.
        '''

        rd_ch = self.config['rd_ch']
        drifts = self.drifts
        rds = []
        rd_imgs = []
        for pos in range(self.images.shape[0]):
            #Load drift correction for image:
            d_y = drifts[pos][1]//ds
            d_x = drifts[pos][2]//ds

            #Read image data:
            img_t = da.max(self.images[pos, 0, rd_ch, ::ds, ::ds, ::ds], axis=0).compute()
            img_o = da.max(self.images[pos, 1, rd_ch, ::ds, ::ds, ::ds], axis=0)

            #Calculate thresholds and label features, filtering to avoid noise.
            thresh = threshold_otsu(img_t)
            labels, n_rds = ndi.label(ndi.median_filter(img_t>thresh, 4))
            bbox = pd.DataFrame(regionprops_table(labels, properties=('label', 'bbox', 'area')))
            bbox['position'] = pos
            bbox = bbox[bbox['area'] > 100]
            bbox['ds'] = ds
            for i, row in bbox.iterrows():
                #Loop over all features and segment them out.
                ymin = row['bbox-0']
                xmin = row['bbox-1']
                ymax = row['bbox-2']
                xmax = row['bbox-3']
                rd_img_t = img_t[ymin:ymax, xmin:xmax]
                
                try:
                    rd_img_o = img_o[ymin-d_y:ymax-d_y, xmin-d_x:xmax-d_x].compute()
                    
                except IndexError: #In case drift correction pushes roi outside image area.
                    ymin = np.clip(ymin-d_y, 0, img_o.shape[1]-1)
                    ymax = np.clip(ymax-d_y, 0, img_o.shape[1]-1)
                    xmin = np.clip(xmin-d_x, 0, img_o.shape[2]-1)
                    xmax = np.clip(xmax-d_x, 0, img_o.shape[2]-1)
                    rd_img_o = img_o[ymin:ymax, xmin:xmax].compute()

                #Expand in case drift correction went outside original image.
                rd_img_o = ip.pad_to_shape(rd_img_o, rd_img_t.shape)
                
                #Calculate new fine scale drift.
                try:
                    rd_img_o = imreg_dft.similarity(rd_img_t, rd_img_o, numiter=10, constraints={
                                                                                 'scale':[1,0],
                                                                                 'angle':[0,10],
                                                                                 'tx':[0,5],
                                                                                 'ty':[0,5]})['timg']
                except IndexError:
                    continue
                new_drift = phase_xcor(rd_img_t, rd_img_o, upsample_factor=4)
                
                #Shift and normalize images for equal comparison
                rd_img_o = ndi.shift(rd_img_o, new_drift)
                rd_img_o = match_histograms(rd_img_o, rd_img_t)
                rd_imgs.append(np.stack([rd_img_t, rd_img_o]))
            rds.append(bbox)
            print(f'Detected {n_rds} replication domains.')
        rds = pd.concat(rds, axis=0).reset_index(drop=True)
        self.rds = rds
        self.rd_imgs = rd_imgs
        return rds, rd_imgs
            

    def compare_imgs(self, kind='nucs', ds=2):
        '''
        Function to compare two (aligned, normalized) image sets according to several metrics.
        Can choose between running on preidentified nuclei ('nucs') or other features ('rds').
        Assumes the appropriate detection function (detect_nucs or detect_rds has been run first)

        Args
        -------
            kind (string, optional): Either 'nucs' or 'rds'.

        Returns
        -------
        Pandas dataframe with comparison results for all images.
        '''
        
        props = []
        if kind == 'nucs':
            imgs = self.z_nuc_imgs
        elif kind == 'rds':
            imgs = self.rd_imgs

        for i, img in enumerate(imgs):
            img_props = {}
            #Calculate and set the comparison metrics in output dataframe.
            ssim_out = comp.comp_ssim(img[0], img[1])
            pcc = comp.comp_pcc_coloc(img[0], img[1])
            mac = comp.comp_mac_coloc(img[0], img[1])
            #orb_ratio = comp.comp_orb_ratio(img[0], img[1])
            #area_ratio, iou = comp.comp_area_iou(img[0], img[1])
            #lbp_score = comp.comp_lbp(img[0], img[1])
            #variance = comp.comp_var(img[0], img[1])
            #skew = comp.comp_skew(img[0], img[1])
            #kurtosis = comp.comp_kurtosis(img[0], img[1])
            
            img_props['id']=i
            img_props['MAC'] = mac
            img_props['PCC'] = pcc
            img_props['SSIM'] = ssim_out
            #img_props['ORB_ratio'] = orb_ratio
            #img_props['Area_ratio']= area_ratio
            #img_props['IOU'] = iou
            #img_props['LBP score'] = lbp_score
            #img_props['Variance ratio'] = variance
            #img_props['Skew ratio'] = skew
            #img_props['Kurtosis ratio'] = kurtosis
        
            props.append(pd.DataFrame(img_props, index=[i]))
            print('Properties calculated.')
        
        props = pd.concat(props)
        if kind == 'nucs':
            self.nuc_metrics = pd.concat([self.nucs, props], axis=1)
        elif kind == 'rds':
            self.rd_metrics =  pd.concat([self.rds, props], axis=1)
        
        return props
    
    def gen_dc_imgs(self, ds=1):
        '''
        Helper function to generate drift corrected images of single nuclei
        and contained features. Currently only coded to take central plane of nucleus
        and overlay with max projection of features.
        #TODO: Make more flexible regarding projections.

        Args:
            ds (int, optional): Downsampling of drift corrected images. Defaults to 1.
        '''

        drifts = self.drifts
        nuc_ch = self.config['nuc_ch']
        rd_ch = self.config['rd_ch']
        imgs = []
        n_rows=len(self.nucs)
        for i, row in self.nucs.iterrows():
            #Loop over single nuclei:
            pos = int(row['position'])
            d_z = drifts[pos][0]
            d_y = drifts[pos][1]
            d_x = drifts[pos][2]
        
            ds_row = row['ds']
            ymin = row['bbox-0']*ds_row
            ymin_o = ymin-d_y
            xmin = row['bbox-1']*ds_row
            xmin_o = xmin-d_x
            ymax = row['bbox-2']*ds_row
            ymax_o = ymax-d_y
            xmax = row['bbox-3']*ds_row
            xmax_o = xmax-d_x

            zmid = row['zmid']*ds_row
            zmid_o = row['zmid_o']*ds_row
            
            #Read image data, drift correct, expand in case boundaries exceed image.
            nuc_img_t = self.images[pos, 0, nuc_ch, zmid, ymin:ymax:ds, xmin:xmax:ds].compute()
            
            try:
                nuc_img_o = self.images[pos, 1, nuc_ch, zmid_o, ymin_o:ymax_o:ds, xmin_o:xmax_o:ds].compute()

            except (IndexError,ValueError): #In case drift correction pushes roi outside image area.
                continue
        
            nuc_img_o = ip.pad_to_shape(nuc_img_o, nuc_img_t.shape)
            
            #Calculate fine scale drift, shift and normalize images.
            try:
                nuc_img_o = imreg_dft.similarity(nuc_img_t, nuc_img_o, numiter=10, constraints={
                                                                                 'scale':[1,0],
                                                                                 'angle':[0,10],
                                                                                 'tx':[0,5],
                                                                                 'ty':[0,5]})['timg']
            except IndexError:
                continue
            new_drift = phase_xcor(nuc_img_t, nuc_img_o, upsample_factor=4)
            nuc_img_o = ndi.shift(nuc_img_o,new_drift)
            nuc_img_o = match_histograms(nuc_img_o, nuc_img_t)

            #Repeat for feature image, except using max projection instead of central slice.
            rd_img_t = da.max(self.images[pos, 0, rd_ch, ::ds, ymin:ymax:ds, xmin:xmax:ds], axis=0).compute()
            try:
                rd_img_o = da.max(self.images[pos, 1, rd_ch, ::ds, ymin-d_y:ymax-d_y:ds, xmin-d_x:xmax-d_x:ds], axis=0).compute()
            except IndexError:
                continue

            rd_img_o = ip.pad_to_shape(rd_img_o, rd_img_t.shape)
            try:
                rd_img_o = imreg_dft.similarity(rd_img_t, rd_img_o, numiter=10, constraints={
                                                                'scale':[1,0],
                                                                'angle':[0,10],
                                                                'tx':[0,5],
                                                                'ty':[0,5]})['timg']
            except IndexError:
                continue
            new_drift = phase_xcor(rd_img_t, rd_img_o, upsample_factor=4)
            rd_img_o = ndi.shift(rd_img_o,new_drift)
            rd_img_o = match_histograms(rd_img_o, rd_img_t)

            img = np.stack([nuc_img_t, nuc_img_o, rd_img_t, rd_img_o])
            imgs.append(img)
            print(f'Generating image {i} of {n_rows}.')
        self.dc_imgs = imgs
        self.save_dc_imgs()

    def save_dc_imgs(self):
        '''
        Convenience function to save drift corrected images.
        '''
        out = self.config['output_folder']+os.sep+self.config['output_name']+'z'+str(self.config['z_offset'])
        exp_dc_imgs = []
        for i, img in enumerate(self.dc_imgs):
            #tiff.imsave(out+'dc_img_'+str(i).zfill(3)+'.tiff', img.astype(np.float32), imagej=True)
            exp_dc_imgs.append(ip.pad_to_shape(img, (img.shape[0], 500, 500)))
        exp_dc_imgs = np.stack(exp_dc_imgs)
        tiff.imsave(out+'dc_img_stack.tiff', exp_dc_imgs.astype(np.float32), imagej=True)

    def save_zarr(self):
        '''
        Function to save images loaded as a 6D PTCZYX dask array into zarr format.
        Will chuck into two last dimensions.
        Also saves a position list of the named positions.
        '''
        zarr_img = da.rechunk(self.images, chunks=(1,1,1,1,-1,-1))
        zarr_img.to_zarr(self.zarr_path, compression='blosc', compression_opts=dict(cname='zstd', clevel=5, shuffle=2))
        self.images = da.from_zarr(self.zarr_path)
        print('Images saved as zarr.')
        
            
    def compare_multi_sc(self, save_dc=False):
        '''
        Running function to calculate similarity metrics on sets of comparison
        images as defined in the config file, saves the results as csv to output folder.
        Can optionally save drift corrected images of each nucleus and feature overlay.

        '''
        if not hasattr(self, 'drifts'):
            self.drift_correction(ds = self.config['initial_dc_ds'])
        self.detect_nuclei(ds= self.config['comp_ds'])
        self.detect_rds(ds= self.config['comp_ds'])
        self.compare_imgs(kind='nucs')
        self.compare_imgs(kind='rds')
        
        out = self.config['output_folder']+os.sep+self.config['output_name']+'z'+str(self.config['z_offset'])+'_'
        self.rd_metrics.to_csv(out+'rd_metrics.csv')
        self.nuc_metrics.to_csv(out+'nuc_metrics.csv')
        if save_dc:
            self.gen_dc_imgs(ds=self.config['dc_ds'])
            self.save_dc_imgs()

