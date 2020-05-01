# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:44:55 2020

@author: ellenberg
"""
import os
import yaml
import numpy as np
import pandas as pd
import image_processing_functions as ip
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from skimage.exposure import match_histograms
from skimage.morphology import dilation, square
import tifffile as tiff


'''
Image processing pipeline usage:
    - Update YAML config file (standard file under Config/)
    - initialize ImageProcess object, e.g.: p=ImageProcess(path_to_config_file)
    - run desired image processing pipelin, e.g. res, imgs = p.compare_multi_sc()
'''

class ImageProcess:
    def __init__(self, config_path):
        '''
        Initialize Image Processing class with config read in from YAML file.
        '''
        
        self.config = ip.load_config(config_path)
        self.config_path = config_path
        
    def reload_config(self):
        self.config = ip.load_config(config_file=self.config_path)
    
    def gen_image_paths(self):
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
        exp_shape = self.config['nuc_exp_bb']
        min_size = int(self.config['min_nuc_vol'])
        plot_ssim = bool(int(self.config['plot_ssim']))
        thresh = bool(int(self.config['threshold_comparison']))
        
        #Load images and detect nuclear masks.
        images = [ip.read_czi_image(path) for path in path_set]
        nuc_labels, nuc_props = ip.detect_nuclei(images[0][nuc_ch], min_size, exp_shape)
        masks = [(nuc_labels == i).astype(int) for i in nuc_props['label'].values]
        #Calculate course drift correction of final image.
        shift=ip.drift_corr_cc(images[0][nuc_ch],ip.pad_to_shape(images[1][nuc_ch],images[0][nuc_ch].shape), upsampling=1, downsampling=8).astype(np.int)
        print(shift)
        nucs=[]
        final_img=[]
        print(nuc_props)
        for ch in analysis_channels:
            
            # Crop out detected individual nuclei (or contents in nuc mask)
            # and expand image to standard size.
            imgs1 = [images[0][ch][(slice(row['lbbox-0'],row['lbbox-3']),
                               slice(row['lbbox-1'],row['lbbox-4']),
                               slice(row['lbbox-2'],row['lbbox-5']))] for i, row in nuc_props.iterrows()]
            imgs1 = [ip.pad_to_shape(img, exp_shape) for img in imgs1]
            
            imgs2 = [images[1][ch][(slice(row['lbbox-0'],row['lbbox-3']),
                               slice(max(0,row['lbbox-1']-shift[1]),min(sy,row['lbbox-4']-shift[1])),
                               slice(max(0,row['lbbox-2']-shift[2]),min(sx,row['lbbox-5']-shift[2])))] for i, row in nuc_props.iterrows()]
            print([img.shape for img in imgs2])
            imgs2 = [ip.pad_to_shape(img, exp_shape) for img in imgs2]
            print([img.shape for img in imgs2])
            #Do a precise drift correction for second image in comparison.
            imgs2_shift = [ip.drift_corr_cc(img[0],img[1], upsampling=4, downsampling=1) for img in zip(imgs1,imgs2)]
            imgs2 = [ndi.shift(img, shift) for img, shift in zip(imgs2,imgs2_shift)]
            
            #Standard filter and image adjustment before comparisons,
            #some comparisons work better when images rescaled to 0-1.
            
            #imgs1 = [ndi.gaussian_filter(img,sigma=2) for img in imgs1]
            imgs1 = [ndi.median_filter(img,size=2) for img in imgs1]
            #imgs2 = [ndi.gaussian_filter(img,sigma=2) for img in imgs2]
            imgs2 = [ndi.median_filter(img,size=2) for img in imgs2]
            
            imgs1 = [img/np.max(img) for img in imgs1]
            imgs2 = [img/np.max(img) for img in imgs2]
            imgs2 = [match_histograms(img[1], img[0]) for img in zip(imgs1, imgs2)]
            
            if thresh:
                #If thresholded comparison is wanted, threshold nuclear mask.
                #from img1 and use this mask on img1 and img2 instead of full images.
                
                nuc_masks = [masks[i][(slice(row['lbbox-0'],row['lbbox-3']),
                                   slice(row['lbbox-1'],row['lbbox-4']),
                                   slice(row['lbbox-2'],row['lbbox-5']))] for i, row in nuc_props.iterrows()]
                nuc_masks = [ip.pad_to_shape(img, exp_shape) for img in nuc_masks]
                nuc_masks = [dilation(img, [square(20)]*10) for img in nuc_masks]
                imgs1 = [img*mask for img,mask in zip(imgs1,nuc_masks)]
                imgs2 = [img*mask for img,mask in zip(imgs2,nuc_masks)]
            
            #Calculate and set the comparison metrics in output dataframe.
            ssim_out, ssim_images = list(zip(*[ip.comp_ssim(img[0], img[1]) for img in zip(imgs1,imgs2)]))
            pcc, mac = list(zip(*[ip.comp_pcc_man_coloc(img[0], img[1]) for img in zip(imgs1,imgs2)])) 
            orb_ratio = [ip.comp_orb_ratio(img[0], img[1]) for img in zip(imgs1,imgs2)]
            area_ratio, iou = list(zip(*[ip.comp_area_iou(img[0], img[1]) for img in zip(imgs1,imgs2)]))
            nuc_props['Channel']=[ch+1]*len(imgs1)
            nuc_props['Title1']=[metadata[0]['Title']]*len(imgs1)
            nuc_props['Title2']=[metadata[1]['Title']]*len(imgs2)
            nuc_props['MAC'] = mac
            nuc_props['PCC'] = pcc
            nuc_props['SSIM'] = ssim_out
            nuc_props['ORB_ratio'] = orb_ratio
            nuc_props['Area_ratio']= area_ratio
            nuc_props['IOU'] = iou
            
            nucs.append(nuc_props.copy())
            #Add filtered, drift corrected, optionally thresholded images
            #to output list of images.
            final_img.append([imgs1, imgs2, ssim_images])

        #Return full dataframes and images containing all individual nuclei.
        return pd.concat(nucs), np.concatenate(final_img, axis=0).astype(np.float32)
            
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
        res, imgs = list(zip(*Parallel(n_jobs=-2, backend='loky')(delayed(self.compare_sc)(path_set) for path_set in image_paths)))
        res = pd.concat(res).reset_index()
        imgs=np.moveaxis(np.concatenate(imgs,axis=1), 0,2)

        output_folder=self.config['output_folder']
        output_name=self.config['output_name']
        res.to_csv(output_folder+os.sep+output_name+'_out.csv')
        tiff.imsave(output_folder+os.sep+output_name+'_imgs.tiff',imgs,imagej=True)
        
        tags=self.config['tags']
        self.metadata=ip.read_czi_meta(image_paths[0][0], tags)
        self.config.update(self.metadata)
        with open(output_folder+os.sep+output_name+'_meta.yaml', 'w') as myfile:
            yaml.safe_dump(self.config, myfile)
        
        return res, imgs
    
    def compare_multi_sr(self):
        config=self.config
        image_paths=self.gen_comp_image_paths()
        sigma=config['const_gauss_sigma']
        cam_px=config['cam_px']
        cam_nm=config['cam_nm']
        grid_px_size=config['grid_px_size']
        output_res=config['output_res']
        dbscan_eps=config['dbscan_eps']
        dbscan_mins=config['dbscan_mins']
        all_props=[]
        image_props={}
        imgs=[]
        print(image_paths)
        for image_set in image_paths:
            locs1=pd.read_csv(image_set[0])
            locs2=pd.read_csv(image_set[1])
            
            locs_db1=ip.clust_dbscan(locs1, dbscan_eps, dbscan_mins, hdb=True)
            print('DBSCAN 1 done')
            locs_db2=ip.clust_dbscan(locs2, dbscan_eps, dbscan_mins, hdb=True)
            print('DBSCAN 2 done')
            render1=ip.render_gauss_const(locs1[locs_db1],sigma,cam_px,cam_nm,grid_px_size, output_res)
            print('Render 1 done')
            render2=ip.render_gauss_const(locs2[locs_db2],sigma,cam_px,cam_nm,grid_px_size, output_res)
            print('Render 2 done')
            shift=ip.drift_corr_cc(np.clip(render1,0,0.5),np.clip(render2,0,0.5), upsampling=2)
            render2=ndi.shift(render2,shift)
            print('DC done')
            pcc, mac = ip.comp_area_iou(render1,render2)
            ssim_out, _ = ip.comp_ssim(render1,render2)
            orb_ratio=ip.comp_orb_ratio(render1,render2)
            area_ratio, iou = ip.comp_area_iou_sr(render1,render2)
            
            print(image_set[0].split('\\')[-1],  pcc, mac, ssim_out, orb_ratio, area_ratio, iou)
            
            image_props['Title1']=image_set[0].split('\\')[-1]
            image_props['Title2']=image_set[1].split('\\')[-1]
            image_props['MAC'] = mac
            image_props['PCC'] = pcc
            image_props['SSIM'] = ssim_out
            image_props['ORB_ratio'] = orb_ratio
            image_props['Area_ratio']= area_ratio
            image_props['IOU'] = iou
            all_props.append(image_props.copy())
            imgs.append([render1,render2])
        
        output_folder=self.config['output_folder']
        output_name=self.config['output_name']
        image_props=pd.DataFrame(all_props)
        image_props.to_csv(output_folder+os.sep+output_name+'_out.csv')
        final_image=np.stack(imgs, axis=0).astype(np.float32)
        tiff.imsave(output_folder+os.sep+output_name+'_imgs.tiff',final_image,imagej=True)
        
        return image_props,final_image
                
            
            