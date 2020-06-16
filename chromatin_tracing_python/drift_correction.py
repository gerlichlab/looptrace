# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 06:18:05 2020

@author: ellenberg
"""

from skimage.registration import phase_cross_correlation
import numpy as np
import scipy.ndimage as ndi
import chromatin_tracing_python.image_processing_functions as ip
from joblib import Parallel, delayed
import yaml
import tifffile as tiff
import os

def drift_corr_cc(t_img, o_img, upsampling=1, downsampling=1):
    '''
    Performs drift correction by cross-correlation.
    Image can be upsampled for sub-pixel accuracy,
    or downsampled (by simple slicing) for speed based on reduced sampling.
    Works in 2D or 3D (ZYX), if 3D with channels (CZYX) then averages shift found for each channel.
    '''
    s = slice(None,None,downsampling)
    if t_img.ndim == 2:
        shift = phase_cross_correlation(t_img[s,s], o_img[s,s], upsample_factor=upsampling, return_error=False)
        shift = shift * [downsampling, downsampling] 
        return shift
    elif t_img.ndim == 3:
        shift = phase_cross_correlation(t_img[:,s,s], o_img[:,s,s], upsample_factor=upsampling, return_error=False)
        shift = shift * [1, downsampling, downsampling] 
        return shift
    elif t_img.ndim == 4:
        shift_all=[]
        for ch in range(t_img.shape[0]):
            shift = phase_cross_correlation(t_img[ch,:,s,s], o_img[ch,:,s,s], upsample_factor=upsampling, return_error=False)
            shift_all.append(shift * [1, downsampling, downsampling])
        shift = np.mean(np.array(shift_all),axis=0)
        return shift
    else:
        raise TypeError('Unsupported number of dimensions.')
            
def drift_corr_fine(template_image, offset_image, upsampling=100):
    '''
    Performs 3d drift correction of one identified bead by cross-correlation.
    Image can be upsampled for sub-pixel accuracy.
    TODO: Increase number of beads/points used by implementing a peak finding algorithm.
    TODO: Adapt to multichannel images.
    '''
    max_proj=np.max(template_image,axis=0)
    max_ind=np.unravel_index(np.argmax(max_proj, axis=None), max_proj.shape)
    s=[slice(ind-16, ind+16) for ind in max_ind]
    shift = phase_cross_correlation(template_image[:,s[0],s[1]], 
                                 offset_image[:,s[0],s[1]], 
                                 upsample_factor=upsampling, 
                                 return_error=False)
    return shift

def drift_corr_multipoint_cc(t_img, o_img, 
                             ch, threshold, min_bead_int, 
                             points=5, upsampling=100):
    t_img_label,num_labels=ndi.label(t_img[ch]>threshold)
    t_img_maxima=np.array(ndi.measurements.maximum_position(t_img[ch], 
                                                   labels=t_img_label, 
                                                   index=range(num_labels)))
    
    t_img_maxima=np.array([m for m in t_img_maxima 
                            if min_bead_int<t_img[ch][tuple(m)]
                            and all(m>8)])
    
    np.random.seed(1)
    rand_points = t_img_maxima[np.random.choice(t_img_maxima.shape[0], size=points), :]
    
    shifts=np.empty_like(rand_points, dtype=np.float32)
    for i, point in enumerate(rand_points):
        s=tuple([slice(ind-8, ind+8) for ind in point])
        shift = phase_cross_correlation(t_img[ch][s], 
                                     o_img[ch][s], 
                                     upsample_factor=upsampling,
                                     return_error=False)
        shifts[i] = shift
    print(shifts)
    return trim_mean(shifts, proportiontocut=0.2, axis=0)

def drift_corr_multipoint_gauss(t_img, o_img, 
                             ch, threshold, bead_int, 
                             points=5, upsampling=100):
    t_img_label,num_labels=ndi.label(t_img[ch]>threshold)
    t_img_maxima=np.array(ndi.measurements.maximum_position(t_img[ch], 
                                                   labels=t_img_label, 
                                                   index=range(num_labels)))
    t_img_maxima=np.array([m for m in t_img_maxima 
                           if (bead_int-200)<t_img[ch][tuple(m)]<(bead_int+200) 
                           and all(m>8)])
    np.random.seed(1)
    rand_points = t_img_maxima[np.random.choice(t_img_maxima.shape[0], size=points), :]
    
    shifts=[]
    for i, point in enumerate(rand_points):
        s=tuple([slice(ind-8, ind+8) for ind in point])
        max_ind_t=list(np.unravel_index(np.argmax(t_img[ch][s], axis=None), 
                                      t_img[ch][s].shape))
        max_ind_o=list(np.unravel_index(np.argmax(o_img[ch][s], axis=None), 
                                      o_img[ch][s].shape))
        gauss_fit_t_img = fitSymmetricGaussian3D(t_img[ch][s], 1, max_ind_t)[0]
        gauss_fit_o_img = fitSymmetricGaussian3D(o_img[ch][s], 1, max_ind_o)[0]

        shift = gauss_fit_t_img[2:5]-gauss_fit_o_img[2:5]
        shifts.append(shift)
    return np.mean(shifts, axis=0)

def drift_shift(t_img, o_img, course_ch, fine_ch):
    # Calculate ZYX drift by FFT CC (subpixel if upsample_factor>1)

    shift=drift_corr_cc(t_img[course_ch], o_img[course_ch], upsampling=1, downsampling=1)
    print('Course shift is', shift)
    
    if fine_ch != -1:
        course_shifted_for_fine=ndi.shift(offset_image[fine_ch],shift,order=0)
        shift_fine=drift_corr_multipoint_cc(t_img[fine_ch], course_shifted_for_fine, upsampling=100)
        print('Fine shift is', shift_fine)
        shift=shift+shift_fine
    #Apply drift correction to stack in all channels, and convert back to uint16
    n_ch=t_img.shape[0]
    shifted_image=np.array([ndi.shift(o_img[i],shift, order=0) for i in range(n_ch)])
    return shifted_image

def drift_corr_image_list(file_list, output_folder, course_ch, fine_ch):
    '''
    Drift corrects in 3D by a distributed register_translation of all files in the list provided. 
    Typically this list is generated by drift_corr_multi or drift_corr_mypic.
    Only CZI files at the moment.
    Input:
        File_list: List of czi images to correct.
        Output_folder:  Output folder to save the drift corrected, merged images. 
        course_ch:      Which channel to use for the main drift correction.
        fine_ch:        If -1 not use. If a channel number is used, will perform a fine
                        drift correction based on finding the brightest feature
                        in the image channel (typically a bead), and doing a highly
                        upsampled cross-correlation.
        
    Output:
        No function output, but saves final driftcorrected series 
        as an imageJ compatible tiff hyperstack in the output folder named according
        to the first image in the list.
    '''
    # Find all CZI files in same dir as template image, exclude template image itself.
    
    #Load template image as CZYX stack.
    template_path = file_list[0]
    template_image = ip.read_czi_image(template_path)
    print('Loaded template: ', template_path)

    #Load images to register
    offset_images = []
    for offset_image_path in file_list[1:]:
        offset_images.append(ip.read_czi_image(offset_image_path))
        print('Loaded image', offset_image_path)
    
    #Prepare for looping along channels for shifting, 
    #and stacking shifted images as hyperstack
            
    output=Parallel(n_jobs=-2)(delayed(drift_shift)(template_image, offset_image, course_ch, fine_ch) for offset_image in offset_images)
    output=np.stack(output,axis=0)
    output=np.concatenate((template_image[np.newaxis,...],output),axis=0)

    #Reshuffle axes to TZCYX order and save as an ImageJ compatible tiff hyperstack.
    output=np.moveaxis(output,1,2)

    #Read metadata from template image and write some more parameters.
    tags=['Title',
          'SizeX',
          'SizeY',
          'SizeZ',
          'SizeC',
          'Model',
          'System',
          'ScalingX',
          'ScalingY',
          'ScalingZ']
    metadata = ip.read_czi_meta(template_path, tags)
    metadata['SizeT']=str(len(file_list))
    metadata['Drift_correction'] = 'Cross correlation'
    filename=metadata['Title'][4:-11]
    print(output_folder+os.sep+filename+'_dc.tif')
    #Save metadata and image file in specified output folder.
    with open(output_folder+os.sep+filename+'_dc.yaml', 'w') as file:
        yaml.safe_dump(metadata, file)
    
    tiff.imsave(output_folder+os.sep+filename+'_dc.tif',output,imagej=True)
    
def drift_corr_mypic(toplevel_folder, output_folder, output_name, course_ch=0, fine_ch=0, filetype='.czi', template='DE_2'):
    pos_folders=[f.path for f in os.scandir(toplevel_folder) if f.is_dir() and 'DE' in f.path]
    for pos in pos_folders:
        image_paths=[pos+os.sep+filename for filename in os.listdir(pos) if filetype in filename and template in filename]
        drift_corr_image_list_h5(image_paths, output_folder, output_name, course_ch, fine_ch)
        

def merge_multipos_lsm(folder):
    image_paths=all_matching_files_in_subfolders(folder,['.lsm'])
    images=[tiff.imread(img) for img in image_paths]
    images=np.concatenate(images, axis=1)
    for pos in range(images.shape[1]):
        tiff.imsave(folder+'_Pos000'+str(pos)+'.tiff',images[pos],imagej=True)
        
def drift_corr_timelapse_tiff(folder, nuc_ch, output_folder):
    image_paths=all_matching_files_in_subfolders(folder,['.tif'])
    for path in image_paths:
        img=tiff.imread(path)
        print('Loaded ', path)
        output = drift_shift_timelapse(img, nuc_ch)
        filename=path.split('\\')[-1].split('.')[0]
        tiff.imsave(output_folder+os.sep+filename+'_DC.tiff',output,imagej=True)
        print('Saved ', output_folder+os.sep+filename+'_DC.tiff')
        
def drift_shift_timelapse(t_image, ch):
    output=np.zeros_like(t_image)
    
    template_image=t_image[0,:,ch,:,:]
    
    def _drift_shift_pl(template_img, offset_img, ch):
        t_img=ndi.sobel(gaussian(template_img,1))
        o_img=ndi.sobel(gaussian(offset_img[:,ch,:,:],1))
        shift=drift_corr_cc(t_img, o_img, upsampling=2, downsampling=1)
        shift=np.insert(shift,1,0)
        shifted_img=ndi.shift(offset_img, shift)
        return shifted_img
    output=Parallel(n_jobs=-1)(delayed(_drift_shift_pl)(template_image,t_image[i],ch) for i in range(1,t_image.shape[0]))
    output=[t_image[0]]+output
    output=np.stack(output,axis=0)
    return output