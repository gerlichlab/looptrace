# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:44:55 2020

@author: ellenberg
"""
import os
import yaml
import numpy as np
import pandas as pd
from chromatin_tracing_python import comparison_functions as comp
from chromatin_tracing_python import image_processing_functions as ip
from chromatin_tracing_python import drift_correction as dc
from chromatin_tracing_python.detect_nuclei import detect_nuclei
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from skimage.exposure import match_histograms
from skimage.morphology import dilation, square
from skimage.registration import phase_cross_correlation
import tifffile as tiff

'''
Image processing pipeline usage:
    - Update YAML config file (standard file under Config/)
    - initialize ImageProcess object, e.g.: p=ImageProcess(path_to_config_file)
    - run desired image processing pipelin, e.g. res, imgs = p.compare_multi_sc()
'''

class Compare:
    def __init__(self, config_path):
        '''
        Initialize Image Processing class with config read in from YAML file.
        '''
        
        self.config = ip.load_config(config_path)
        self.config_path = config_path
        self.save_images = bool(int(self.config['save_comparison_image']))
        
    def reload_config(self):
        self.config = ip.load_config(config_file=self.config_path)
    
    def gen_image_paths(self):
        #TODO Deprecate this function, gen_comp_image_paths has superseeded this.

        config=self.config
        input_folders=config['folders']
        n_dirs = len(input_folders)
        template=config['template']
        filetype=config['filetype']
        print(config)
        all_folders=zip(*[[f.path for f in os.scandir(folder) if f.is_dir()] for folder in input_folders])
        paths_=[folder+os.sep+filename for folders in all_folders for folder in folders for filename in os.listdir(folder) 
                             if filetype in filename and all(s in filename for s in template)]
        image_paths=list(zip(*([iter(paths_)]*n_dirs)))
        return image_paths
    
    def gen_comp_image_paths(self):
        config=self.config
        print(config)
        template1=config['template_comp_1']+config['filetype']
        template2=config['template_comp_2']+config['filetype']
        input1=config['comp_folder_1']
        input2=config['comp_folder_2']
        
        images1=ip.all_matching_files_in_subfolders(input1,template1)
        images2=ip.all_matching_files_in_subfolders(input2,template2)
        
        return list(zip(images1,images2))

    def compare_sc(self, path_set):
        '''
        Function to compare two images according to several metrics.
        Can be multiple channels and assumes a nuclear mask in one defined ch.

        Parameters
        ----------
        path_set : A tuple of two paths to images (czi images atm).

        Returns
        -------
        Pandas dataframe with properties of each object in each channel.
        Combined image of all objects with different images/comparisons in 
        channels and different objects in T.

        '''
        
        #Read parameters needed from config and metadata.
        tags = self.config['tags']
        metadata = [ip.read_czi_meta(path, tags) for path in path_set]
        sz,sy,sx = int(metadata[0]['SizeZ']),int(metadata[0]['SizeY']),int(metadata[0]['SizeX'])
        analysis_channels = self.config['analysis_channels']
        nuc_ch = int(self.config['nuc_ch'])
        exp_shape = tuple(self.config['nuc_exp_bb'])
        min_size = int(self.config['min_nuc_vol'])
        plot_ssim = bool(int(self.config['plot_ssim']))
        thresh = bool(int(self.config['threshold_comparison']))
        output_folder=self.config['output_folder']
        output_name=metadata[0]['Title'].split(':')[-1]+self.config['output_name']
        output_name+='_'+path_set[0].split('_')[-4]
        
        #Load images and detect nuclear masks.
        images = [ip.read_czi_image(path) for path in path_set]
        print('Loaded images:', path_set)
        print('Image shape:', images[0].shape)
        
        nuc_labels, nuc_props = detect_nuclei(images[0][nuc_ch], min_size, exp_shape)
        print('Detected {} nuclei.'.format(len(nuc_props)))
        
        #Handle case if no nuclei are detected.
        if len(nuc_props) == 0:
            print('No nuclei detected, returning empty.')
            return pd.DataFrame()

        masks = [(nuc_labels == i).astype(int) for i in nuc_props['label'].values]
        #Calculate course drift correction of final image.
        shift=dc.drift_corr_cc(images[0][nuc_ch],
                               ip.pad_to_shape(images[1][nuc_ch],
                                               images[0][nuc_ch].shape), 
                               upsampling=1, 
                               downsampling=8).astype(np.int)
        print('Found course shift:', shift)
        nucs=[]
        final_img=[]
        print(nuc_props)
        for ch in analysis_channels:
            print('Processing ch ', ch)
            # Crop out detected individual nuclei (or contents in nuc mask)
            # and expand image to standard size.
            imgs1 = [images[0][ch][(slice(int(row['lbbox-0']),int(row['lbbox-3'])),
                               slice(int(row['lbbox-1']),int(row['lbbox-4'])),
                               slice(int(row['lbbox-2']),int(row['lbbox-5'])))] for 
                                    i, row in nuc_props.iterrows()]
            imgs1 = [ip.pad_to_shape(img, exp_shape) for img in imgs1]
            
            imgs2 = [images[1][ch][(slice(int(row['lbbox-0']),int(row['lbbox-3'])),
                               slice(max(0,int(row['lbbox-1'])-shift[1]),min(sy,int(row['lbbox-4'])-shift[1])),
                               slice(max(0,int(row['lbbox-2'])-shift[2]),min(sx,int(row['lbbox-5']-shift[2]))))] for 
                                    i, row in nuc_props.iterrows()]
            
            imgs2 = [ip.pad_to_shape(img, exp_shape) for img in imgs2]
            print('Images cropped and padded.')
            #Do a precise drift correction for second image in comparison.
            imgs2_shift = [dc.drift_corr_cc(img[0],img[1], upsampling=4, downsampling=1) for 
                           img in zip(imgs1,imgs2)]
            imgs2 = [ndi.shift(img, shift) for img, shift in zip(imgs2,imgs2_shift)]
            print('Fine drift correction done.')
            #Standard filter and image adjustment before comparisons,
            #some comparisons work better when images rescaled to 0-1.
            
            #imgs1 = [ndi.gaussian_filter(img,sigma=2) for img in imgs1]
            #imgs1 = [ndi.median_filter(img,size=2) for img in imgs1]
            #imgs2 = [ndi.gaussian_filter(img,sigma=2) for img in imgs2]
            #imgs2 = [ndi.median_filter(img,size=2) for img in imgs2]
            
            imgs1 = [(img/np.max(img)).astype(np.float32) for img in imgs1]
            imgs2 = [(img/np.max(img)).astype(np.float32) for img in imgs2]
            imgs2 = [match_histograms(img[1], img[0]).astype(np.float32) 
                    for img in zip(imgs1, imgs2)]
            print('Images rescaled.')
            
            if thresh:
                #If thresholded comparison is wanted, threshold nuclear mask.
                #from img1 and use this mask on img1 and img2 instead of full images.
                
                nuc_masks = [masks[i][(slice(int(row['lbbox-0']),int(row['lbbox-3'])),
                                   slice(int(row['lbbox-1']),int(row['lbbox-4'])),
                                   slice(int(row['lbbox-2']),int(row['lbbox-5'])))] for 
                                       i, row in nuc_props.iterrows()]
                nuc_masks = [ip.pad_to_shape(img, exp_shape) for img in nuc_masks]
                nuc_masks = [dilation(img, [square(20)]*10) for img in nuc_masks]
                imgs1 = [(img*mask).astype(np.float32) for img,mask in zip(imgs1,nuc_masks)]
                imgs2 = [(img*mask).astype(np.float32) for img,mask in zip(imgs2,nuc_masks)]
            
            #Calculate and set the comparison metrics in output dataframe.
            ssim_out, ssim_images = list(zip(*[comp.comp_ssim(img[0], img[1]) for img in zip(imgs1,imgs2)]))
            del ssim_images
            pcc, mac = list(zip(*[comp.comp_pcc_man_coloc(img[0], img[1]) for img in zip(imgs1,imgs2)])) 
            orb_ratio = [comp.comp_orb_ratio(img[0], img[1]) for img in zip(imgs1,imgs2)]
            area_ratio, iou = list(zip(*[comp.comp_area_iou(img[0], img[1]) for img in zip(imgs1,imgs2)]))
            lbp_score = [comp.comp_lbp(img[0], img[1]) for img in zip(imgs1,imgs2)]
            variance = [comp.comp_var(img[0], img[1]) for img in zip(imgs1,imgs2)]
            skew = [comp.comp_skew(img[0], img[1]) for img in zip(imgs1,imgs2)]
            kurtosis = [comp.comp_kurtosis(img[0], img[1]) for img in zip(imgs1,imgs2)]
            
            nuc_props['Channel']=[ch+1]*len(imgs1)
            nuc_props['Title1']=[metadata[0]['Title']]*len(imgs1)
            nuc_props['Title2']=[metadata[1]['Title']]*len(imgs2)
            nuc_props['MAC'] = mac
            nuc_props['PCC'] = pcc
            nuc_props['SSIM'] = ssim_out
            nuc_props['ORB_ratio'] = orb_ratio
            nuc_props['Area_ratio']= area_ratio
            nuc_props['IOU'] = iou
            nuc_props['LBP score'] = lbp_score
            nuc_props['Variance ratio'] = variance
            nuc_props['Skew ratio'] = skew
            nuc_props['Kurtosis ratio'] = kurtosis
            
            nucs.append(nuc_props.copy())
            #Add filtered, drift corrected, optionally thresholded images
            #to output list of images.
            final_img.append([imgs1, imgs2])
            print('Properties calculated.')
        #Return full dataframes and images containing all individual nuclei.
        del images
        if self.save_images:
            final_img=np.concatenate(final_img, axis=0)
            print('Final image shape', final_img.shape)
            final_img=np.moveaxis(final_img, 0, 2)
            tiff.imsave(output_folder+os.sep+output_name+'_nucs_comp.tiff',final_img,imagej=True)
            print('Saved ', output_folder+os.sep+output_name+'_nucs_comp.tiff')
        print('Processing done on:', path_set)
        return pd.concat(nucs)
    
    def merge_dc_matched(self):
        #IN PROGRESS
        image_paths=self.gen_comp_image_paths()
        output_folder=self.config['output_folder']
        
        dc_channel=self.config['drift_correction_channel']
        
        for path_set in image_paths:
            images = [ip.read_czi_image(path) for path in path_set]
            print('Loaded images:', path_set)
            print('Image shape:', images[0].shape)
            images[0] = ip.pad_to_shape(images[0], images[1].shape)
            print('New image shape:', images[0].shape)
            shift = phase_cross_correlation(images[0][dc_channel], images[1][dc_channel],
                                            upsample_factor=5, return_error=False)
            shift=[0]+list(shift)
            print(shift)
            images[1] = ndi.shift(images[1], shift, order=0)
            
            print(images[1].shape, images[0].shape)
            output=np.concatenate(images, axis=0)
            output=np.moveaxis(output, 0, 1)
            print(output.shape)
            output_name= self.config['output_name']+'_'+path_set[0].split('_')[-4]
            out_path=output_folder+os.sep+output_name+'_comp.tiff'
            tiff.imsave(out_path,output,imagej=True)
            print('Saved image in', out_path)
        
            
    def compare_multi_sc(self):
        '''
        Running function to calculate similarity metrics on sets of comparison
        images from different folders, and saving output data.

        Returns
        -------
        res : Pandas dataframe of metric calculated from all images.
        imgs : Combined images.

        '''
        
        image_paths=self.gen_comp_image_paths()
        res = Parallel(n_jobs=1, backend='loky')(delayed(self.compare_sc)(path_set) for path_set in image_paths)
        res = pd.concat(res).reset_index()
        
        output_folder=self.config['output_folder']
        output_name=self.config['output_name']
        res.to_csv(output_folder+os.sep+output_name+'_out.csv')
        
        tags=self.config['tags']
        self.metadata=ip.read_czi_meta(image_paths[0][0], tags)
        self.config.update(self.metadata)
        with open(output_folder+os.sep+output_name+'_meta.yaml', 'w') as myfile:
            yaml.safe_dump(self.config, myfile)
        
        return res
    
                
            
            