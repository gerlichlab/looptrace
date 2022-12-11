
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from looptrace.ImageHandler import ImageHandler
from looptrace import image_io
from looptrace import image_processing_functions as ip
import os
import argparse
import itk
import numpy as np
import numpy as np
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract experimental PSF from bead images.')
    parser.add_argument("config_path", help="Experiment config file path")
    parser.add_argument("image_path", help="Path to folder with images to read.")
    parser.add_argument('template_image_name', help='Name of images in image_path to use as template')
    parser.add_argument('template_image_frame', help='Channel of images in image_path to use as template')
    parser.add_argument('template_image_channel', help='Channel of images in image_path to use as template')
    parser.add_argument('moving_image_name', help='Name of images in image_path to use as template')
    parser.add_argument('moving_image_channel', help='Channel of images in image_path to use as template')
    parser.add_argument('downsample', help='Factor to downsample by.', default='4')
    parser.add_argument('elastix_config_file', help='Path to elastix config file.')
    parser.add_argument('cells_crop', help='Toggle to additionally center and crop cells images.')
    parser.add_argument("--image_save_path", help="(Optional): Path to folder to save images to.", default=None)
    
    args = parser.parse_args()
    H = ImageHandler(config_path=args.config_path, image_path=args.image_path, image_save_path=args.image_save_path)
    positions = H.image_lists[args.template_image_name]
    if args.cells_crop == 'on':
        crop_size = H.config['cells_crop_size']
    #ds = int(args.downsample)

    try:
        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"]) #Split datasets by positions if running across SLURM cluster.
        positions = [positions[array_id]]
    except KeyError:
        array_id = None

    print('Running elastix drift correction')
    for pos in positions:
        print('Drift correcting position ', pos)
        pos_id = H.image_lists[args.template_image_name].index(pos) #If running in cluster, get correct position.
        #Get the template image from the image handler.
        fixed = np.array(H.images[args.template_image_name][pos_id][int(args.template_image_frame), int(args.template_image_channel)])
        if args.cells_crop == 'on':
            fixed = ip.center_crop_cells(fixed, crop_size)
        fixed = itk.GetImageFromArray(fixed.astype(np.float32))

        print('Loaded fixed image of shape ', fixed.shape)
        img_shape = H.images[args.moving_image_name][pos_id].shape[0:2]+fixed.shape #Find overall shape of output zarr array (same as all the moving images to be registered)
    
        #Create OME-ZARR store for position.
        z = image_io.create_zarr_store(path=H.image_save_path+os.sep+args.moving_image_name+'_registered',
                    name = args.moving_image_name+'_registered', 
                    pos_name = pos,
                    shape = img_shape, 
                    dtype = np.uint16,  
                    chunks = (1,1,1,img_shape[-2], img_shape[-1]))

        for frame_id in tqdm.tqdm(range(H.images[args.moving_image_name][pos_id].shape[0])): #Loop trhough all the frames to be registered.
            
            #Load all channels of the moving image into RAM (since we will be transforming and saving all channels)
            moving_allch = np.array(H.images[args.moving_image_name][pos_id][frame_id])

            #Find center of the group of cells from the registration image, and crop all channels to center.
            if args.cells_crop == 'on':
                center = ip.find_center_of_cells(moving_allch[int(args.moving_image_channel)])
                moving_allch = np.stack([ip.center_crop_cells(ch_img, crop_size, center) for ch_img in moving_allch])

            #Read the images with elastix and specify the data-type (data must be np.float32 for conversion to itk format):
            moving = itk.GetImageFromArray(moving_allch[int(args.moving_image_channel)].astype(np.float32))
            print('Loaded moving image of shape ', moving.shape)

            # Create a parameter object and read the parameters from a text file
            parameter_object = itk.ParameterObject.New()
            parameter_object.ReadParameterFile(args.elastix_config_file)

            #Create the registration object, load parameters, set further options and run registration, extracting the parameters in the end.
            elastix_object = itk.ElastixRegistrationMethod.New(fixed, moving)
            elastix_object.SetParameterObject(parameter_object)
            elastix_object.SetLogToConsole(False)
            elastix_object.SetNumberOfThreads(4)
            elastix_object.UpdateLargestPossibleRegion()
            result_transform = elastix_object.GetTransformParameterObject()

            #Apply transformation to all channels, and write each of the transformed images to the correct position in the zarr file:
            for ch in range(moving_allch.shape[0]):
                moving = itk.GetImageFromArray(moving_allch[ch].astype(np.float32))
                transformix_object = itk.Elastix.TransformixFilter.New(moving)
                transformix_object.SetTransformParameterObject(result_transform)
                transformix_object.UpdateLargestPossibleRegion()
                z[frame_id, ch] = np.asarray(transformix_object.GetOutput()).astype(np.uint16)
            print('Saved transformed position ', pos)

