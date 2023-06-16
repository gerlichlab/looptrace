# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg
"""

from scipy import ndimage as ndi
from looptrace import image_processing_functions as ip
from looptrace import image_io
import pandas as pd
import numpy as np
import random
from skimage.measure import regionprops_table
import tqdm
import os

class SpotPicker:
    def __init__(self, image_handler, array_id = None):
        self.image_handler = image_handler
        self.config = image_handler.config
        self.images = self.image_handler.images[self.config['spot_input_name']]
        self.pos_list = self.image_handler.image_lists[self.config['spot_input_name']]
        self.roi_path = self.image_handler.out_path+self.config['spot_input_name']+'_rois.csv'
        self.dc_roi_path = self.image_handler.out_path+self.config['spot_input_name']+'_dc_rois.csv'
        self.array_id = array_id
        if self.array_id is not None:
            self.pos_list = [self.pos_list[int(self.array_id)]]
            self.roi_path = self.image_handler.out_path+self.config['spot_input_name']+'_rois.csv'[:-4]+'_'+str(self.array_id).zfill(4)+'.csv'
        
    def rois_from_spots(self, preview_pos=None):
        '''
        Autodetect ROIs from spot images using a manual threshold defined in config.
        
        Returns
        ---------
        A pandas DataFrame with the bounding boxes and identifyer of the detected ROIs. 
        NB: Null (None) returned if no spots are found.
        '''

        #Read parameters from config.
        
        spot_ch = self.config['spot_ch']
        if not isinstance(spot_ch, list):
            spot_ch = [spot_ch]
        spot_frame = self.config['spot_frame']
        if not isinstance(spot_frame, list):
            spot_frame = [spot_frame]
        
        try:
            subtract_beads = self.config['subtract_crosstalk']
            crosstalk_ch = self.config['crosstalk_ch']
        except KeyError: #Legacy config.
            subtract_beads = False

        try:
            min_dist = self.config['min_spot_dist']
        except KeyError: #Legacy config.
            min_dist = None

        try:
            filter_nucs = self.config['spot_in_nuc']
            if 'nuc_images_drift_correction' in self.image_handler.tables:
                nuc_drifts = self.image_handler.tables[self.config['nuc_input_name']+'_drift_correction']
                nuc_target_frame = self.config['nuc_ref_frame']
                spot_drifts = self.image_handler.tables[self.config['spot_input_name']+'_drift_correction']
            else:
                nuc_drifts = None
                nuc_target_frame = None
                spot_drifts = None
        except KeyError: #Legacy config.
            filter_nucs = False
        

        spot_threshold = self.config['spot_threshold']
        if not isinstance(spot_threshold, list):
            spot_threshold = [spot_threshold]*len(spot_frame)
        spot_ds = self.config['spot_downsample']
        
        try:
            detect_method = self.config['detection_method']
            if detect_method == 'intensity':
                detect_func = ip.detect_spots_int
                print('Using intensity based spot detection with threshold ', spot_threshold)
            else:
                detect_func = ip.detect_spots
                print('Using DoG based spot detection with threshold ',  spot_threshold)
        except KeyError: #Legacy config.
            detect_func = ip.detect_spots
            detect_method = 'dog'
            print('Using DoG based spot detection.')

        center_spots = (lambda df: ip.roi_center_to_bbox(df, roi_size = np.array(self.config['roi_image_size']) // spot_ds)) if detect_method != 'intensity' else (lambda df: df)
        
        #Loop through the imaging positions.
        all_rois = []
        if preview_pos is not None:
            for i, frame in enumerate(spot_frame):
                for j, ch in enumerate(spot_ch):
                    print(f'Preview spot detection in position {preview_pos}, frame {frame} with threshold {spot_threshold[i]}.')
                    pos_index = self.image_handler.image_lists[self.config['spot_input_name']].index(preview_pos)
                    img = self.images[pos_index][frame, ch, ::spot_ds, ::spot_ds, ::spot_ds].compute()

                    if subtract_beads:
                        bead_img = self.images[pos_index][frame, crosstalk_ch, ::spot_ds, ::spot_ds, ::spot_ds].compute()
                        img, orig = ip.subtract_crosstalk(bead_img, img, threshold=self.config['bead_threshold'])

                    spot_props, filt_img = detect_func(img, spot_threshold[i], min_dist = min_dist)
                    spot_props['position'] = preview_pos
                    spot_props = spot_props.reset_index().rename(columns={'index':'roi_id_pos'})

                    spot_props = center_spots(spot_props)
                    
                    roi_points, _ = ip.roi_to_napari_points(spot_props, position=preview_pos)
                    try:
                        ip.napari_view(np.stack([filt_img, img, orig]), axes = 'CZYX', points=roi_points, downscale=1, name = ['DoG', 'Subtracted', 'Original'])
                    except NameError:
                        ip.napari_view(np.stack([filt_img, img]), axes = 'CZYX', points=roi_points, downscale=1, name = ['DoG', 'Original'])

            return
        
        for position in tqdm.tqdm(self.pos_list):
            pos_index = self.image_handler.image_lists[self.config['spot_input_name']].index(position)
            for i, frame in tqdm.tqdm(enumerate(spot_frame)):
                for j, ch in enumerate(spot_ch):
                #print(f'Detecting spots in position {position}, frame {frame}, ch {ch}.',  end=' ')
                    img = self.images[pos_index][frame, ch, ::spot_ds, ::spot_ds, ::spot_ds].compute()
                    if subtract_beads:
                        bead_img = self.images[pos_index][frame, crosstalk_ch, ::spot_ds, ::spot_ds, ::spot_ds].compute()
                        img, _ = ip.subtract_crosstalk(bead_img, img, threshold=self.config['bead_threshold'])
                    spot_props, _ = detect_func(img, spot_threshold[i], min_dist = min_dist)

                    spot_props = center_spots(spot_props)
                    
                    spot_props[['z_min', 'y_min', 'x_min', 'z_max', 'y_max', 'x_max', 'zc', 'yc', 'xc']] = spot_props[['z_min', 'y_min', 'x_min', 'z_max', 'y_max', 'x_max', 'zc', 'yc', 'xc']]*spot_ds
                    
                    spot_props['position'] = position
                    spot_props['frame'] = frame
                    spot_props['ch'] = ch
                    all_rois.append(spot_props)
        output = pd.concat(all_rois)

        n_spots_init = len(output)
        print(f'Found {len(output)} spots.')
        if n_spots_init == 0:
            print("WARNING -- Since no spots were found, no spots file(s) will be written.")
            return

        if filter_nucs:
            if 'nuc_masks' not in self.image_handler.images:
                print('No nuclei mask images found, cannot filter.')
            else:
                print('Filtering in nuclei.')
                if self.array_id is None:
                    nuc_masks = [a[0,0] for a in self.image_handler.images['nuc_masks']]
                else:
                    nuc_masks = [self.image_handler.images['nuc_masks'][pos_index][0,0]]

                output = ip.filter_rois_in_nucs(output, nuc_masks, self.pos_list, new_col='nuc_label', nuc_drifts=nuc_drifts, nuc_target_frame=nuc_target_frame, spot_drifts = spot_drifts)
                output = output[output['nuc_label'] > 0 ]
            
            if 'nuc_classes' in self.image_handler.images:
                print('Assigning nucleus classes.')
                if self.array_id is None:
                    nuc_classes = [a[0,0] for a in self.image_handler.images['nuc_classes']]
                else:
                    nuc_classes = [self.image_handler.images['nuc_classes'][pos_index][0,0]]
                output = ip.filter_rois_in_nucs(output, nuc_classes, self.pos_list, new_col='nuc_class', nuc_drifts=nuc_drifts, nuc_target_frame=nuc_target_frame, spot_drifts = spot_drifts)

        output=output.reset_index().rename(columns={'index':'roi_id_pos'})
        if detect_method == 'dog':
            rois = ip.roi_center_to_bbox(output, roi_size = tuple(self.config['roi_image_size']))
        else:
            rois = output

        rois = rois.sort_values(['position', 'frame'])
        rois.to_csv(self.roi_path)
        self.image_handler.load_tables()
        print(f'Filtering complete, {len(rois)} ROIs after filtering.')
        
        return rois

    def rois_from_beads(self):
        print('Detecting bead ROIs for tracing.')
        all_rois = []
        n_fields = self.config['bead_trace_fields']
        n_beads = self.config['bead_trace_number']
        for pos in tqdm.tqdm(random.sample(self.pos_list, k=n_fields)):
            pos_index = self.image_handler.image_lists[self.config['spot_input_name']].index(pos)
            ref_frame = self.config['bead_reference_frame']
            ref_ch = self.config['bead_ch']
            threshold = self.config['bead_threshold']
            min_bead_int = self.config['min_bead_intensity']

            t_img = self.images[pos_index][ref_frame, ref_ch].compute()
            t_img_label,num_labels = ndi.label(t_img>threshold)
            #t_img_maxima=np.array(ndi.measurements.maximum_position(t_img, 
            #                                            labels=t_img_label, 
            #                                            index=np.random.choice(np.arange(1,num_labels), size=n_points*2)))
            
            spot_props = pd.DataFrame(regionprops_table(t_img_label, t_img, properties=('label', 'centroid', 'max_intensity')))
            spot_props = spot_props.query('max_intensity > @min_bead_int').sample(n=n_beads, random_state=1)
            
            spot_props.drop(['label'], axis=1, inplace=True)
            spot_props.rename(columns={'centroid-0': 'zc',
                                        'centroid-1': 'yc',
                                        'centroid-2': 'xc',
                                        'index':'roi_id_pos'},
                                        inplace = True)

            spot_props['position'] = pos
            spot_props['frame'] = ref_frame
            spot_props['ch'] = ref_ch
            print('Detected beads in position', pos, spot_props)
            all_rois.append(spot_props)
        
        output = pd.concat(all_rois)
        output=output.reset_index().rename(columns={'index':'roi_id'})
        rois = ip.roi_center_to_bbox(output, roi_size = tuple(self.config['roi_image_size']))

        self.image_handler.bead_rois = rois
        rois.to_csv(self.roi_path+'_beads.csv')
        self.image_handler.load_tables()
        return rois
    '''
    def refilter_rois(self):
        #TODO needs updating to new table syntax.
        if self.config['detection_method'] == 'dog':
            self.image_handler.roi_table = ip.roi_center_to_bbox(self.image_handler.roi_table.copy(), roi_size = tuple(self.config['roi_image_size']))
        
        self.image_handler.roi_table[['z_min', 'y_min', 'x_min', 'z_max', 'y_max', 'x_max', 'zc', 'yc', 'xc']] = self.image_handler.roi_table[['z_min', 'y_min', 'x_min', 'z_max', 'y_max', 'x_max', 'zc', 'yc', 'xc']].round().astype(int)

        if 'nuc_images' not in self.image_handler.images:
            print('Please generate nuclei images first.')
        if 'nuc_masks' in self.image_handler.images:
            self.image_handler.roi_table = ip.filter_rois_in_nucs(self.image_handler.roi_table.copy(), np.array(self.image_handler.images['nuc_masks']), self.pos_list, new_col='nuc_label', drifts = self.image_handler.tables['nuc_images_drift_correction'], target_frame=self.config['nuc_ref_frame'])
            
        if 'nuc_classes' in self.image_handler.images:
            self.image_handler.roi_table = ip.filter_rois_in_nucs(self.image_handler.roi_table.copy(), np.array(self.image_handler.images['nuc_classes']), self.pos_list, new_col='nuc_class', drifts = self.image_handler.tables['nuc_images_drift_correction'], target_frame=self.config['nuc_ref_frame'])

        print('ROIs (re)assigned to nuclei.')
        self.image_handler.roi_table.to_csv(self.roi_path)
        self.image_handler.load_tables()
    '''
    def make_dc_rois_all_frames(self):
        #Precalculate all ROIs for extracting spot images, based on identified ROIs and precalculated drifts between time frames.
        print('Generating list of all ROIs for tracing:')
        #positions = sorted(list(self.roi_table.position.unique()))

        all_rois = []
        for i, roi in tqdm.tqdm(self.image_handler.tables[self.config['spot_input_name']+'_rois'].iterrows(), total=len(self.image_handler.tables[self.config['spot_input_name']+'_rois'])):
            pos = roi['position']
            pos_index = self.image_handler.image_lists[self.config['spot_input_name']].index(pos)#positions.index(pos)
            dc_pos_name = self.image_handler.image_lists[self.config['reg_input_moving']][pos_index]
            sel_dc = self.image_handler.tables[self.config['spot_input_name']+'_drift_correction'].query('position == @dc_pos_name')
            ref_frame = roi['frame']
            ch = roi['ch']
            ref_offset = sel_dc.query('frame == @ref_frame')
            Z, Y, X = self.images[pos_index][0,ch].shape[-3:]
            for j, dc_frame in sel_dc.iterrows():
                z_drift_course = int(dc_frame['z_px_course']) - int(ref_offset['z_px_course'])
                y_drift_course = int(dc_frame['y_px_course']) - int(ref_offset['y_px_course'])
                x_drift_course = int(dc_frame['x_px_course']) - int(ref_offset['x_px_course'])

                z_min = max(roi['z_min'] - z_drift_course, 0)
                z_max = min(roi['z_max'] - z_drift_course, Z)
                y_min = max(roi['y_min'] - y_drift_course, 0)
                y_max = min(roi['y_max'] - y_drift_course, Y)
                x_min = max(roi['x_min'] - x_drift_course, 0)
                x_max = min(roi['x_max'] - x_drift_course, X)

                pad_z_min = abs(min(0,z_min))
                pad_z_max = abs(max(0,z_max-Z))
                pad_y_min = abs(min(0,y_min))
                pad_y_max = abs(max(0,y_max-Y))
                pad_x_min = abs(min(0,x_min))
                pad_x_max = abs(max(0,x_max-X))

                #print('Appending ', s)
                all_rois.append([pos, pos_index, roi.name, dc_frame['frame'], ref_frame, ch, 
                                z_min, z_max, y_min, y_max, x_min, x_max, 
                                pad_z_min, pad_z_max, pad_y_min, pad_y_max, pad_x_min, pad_x_max,
                                z_drift_course, y_drift_course, x_drift_course, 
                                dc_frame['z_px_fine'], dc_frame['y_px_fine'], dc_frame['x_px_fine']])

        self.all_rois = pd.DataFrame(all_rois, columns=['position', 'pos_index', 'roi_id', 'frame', 'ref_frame', 'ch', 
                                'z_min', 'z_max', 'y_min', 'y_max', 'x_min', 'x_max',
                                'pad_z_min', 'pad_z_max', 'pad_y_min', 'pad_y_max', 'pad_x_min', 'pad_x_max', 
                                'z_px_course', 'y_px_course', 'x_px_course',
                                'z_px_fine', 'y_px_fine', 'x_px_fine'])
        self.all_rois = self.all_rois.sort_values(['roi_id','frame'])
        print(self.all_rois)
        self.all_rois.to_csv(self.dc_roi_path)
        self.image_handler.load_tables()

    def extract_single_roi_img(self, single_roi):
        #Function to extract single ROI lazily without loading entire stack in RAM.
        #Depending on chunking of original data can be more or less performant.

        roi_image_size = tuple(self.config['roi_image_size'])
        p = single_roi['pos_index']
        t = single_roi['frame']
        c = single_roi['ch']
        z = slice(single_roi['z_min'], single_roi['z_max'])
        y = slice(single_roi['y_min'], single_roi['y_max'])
        x = slice(single_roi['x_min'], single_roi['x_max'])
        pad = ( (single_roi['pad_z_min'], single_roi['pad_z_max']),
                (single_roi['pad_y_min'], single_roi['pad_y_max']),
                (single_roi['pad_x_min'], single_roi['pad_x_max']))

        try:
            roi_img = np.array(self.images[p][t, c, z, y, x])

            #If microscope drifted, ROI could be outside image. Correct for this:
            if pad != ((0,0),(0,0),(0,0)):
                #print('Padding ', pad)
                roi_img = np.pad(roi_img, pad, mode='edge')

        except ValueError: # ROI collection failed for some reason
            roi_img = np.zeros(roi_image_size, dtype=np.float32)

        #print(p, t, c, z, y, x)
        return roi_img  #{'p':p, 't':t, 'c':c, 'z':z, 'y':y, 'x':x, 'img':roi_img}

    def extract_single_roi_img_inmem(self, single_roi, images):
        # Function for extracting a single cropped region defined by ROI from a larger 3D image.
        z = slice(single_roi['z_min'], single_roi['z_max'])
        y = slice(single_roi['y_min'], single_roi['y_max'])
        x = slice(single_roi['x_min'], single_roi['x_max'])
        pad = ( (single_roi['pad_z_min'], single_roi['pad_z_max']),
                (single_roi['pad_y_min'], single_roi['pad_y_max']),
                (single_roi['pad_x_min'], single_roi['pad_x_max']))

        try:
            roi_img = np.array(images[z, y, x])

            #If microscope drifted, ROI could be outside image. Correct for this:
            if pad != ((0,0),(0,0),(0,0)):
                roi_img = np.pad(roi_img, pad, mode='edge')

        except ValueError: # ROI collection failed for some reason
            roi_img = np.zeros((np.abs(z.stop-z.start), np.abs(y.stop-y.start), np.abs(x.stop-x.start)), dtype=np.float32)

        return roi_img  

    def gen_roi_imgs_inmem(self):
        from numpy.lib.format import open_memmap
        # Load full stacks into memory to extract spots.
        # Not the most elegant, but depending on the chunking of the original data it is often more performant than loading subsegments.

        rois = self.image_handler.tables[self.config['spot_input_name']+'_dc_rois']
        if self.array_id is not None:
            pos_name = self.image_handler.image_lists[self.config['spot_input_name']][self.array_id]
            rois = rois[rois.position == pos_name]


        if not os.path.isdir(self.image_handler.image_save_path+os.sep+'spot_images_dir'):
            os.mkdir(self.image_handler.image_save_path+os.sep+'spot_images_dir')

        for pos, pos_group in tqdm.tqdm(rois.groupby('position')):
            pos_index = self.image_handler.image_lists[self.config['spot_input_name']].index(pos)
            #print(full_image.shape)
            f_id = 0
            n_frames = len(pos_group.frame.unique())
            #print(n_frames)
            for frame, frame_group in tqdm.tqdm(pos_group.groupby('frame')):
                for ch, ch_group in frame_group.groupby('ch'):
                    image_stack = np.array(self.images[pos_index][int(frame), int(ch)])
                    for i, roi in ch_group.iterrows():
                        roi_img = self.extract_single_roi_img_inmem(roi, image_stack).astype(np.uint16)
                        fn = self.image_handler.image_save_path+os.sep+'spot_images_dir'+os.sep+str(pos)+'_'+str(roi['roi_id']).zfill(5)+'.npy'
                        if f_id == 0:
                            arr = open_memmap(fn, mode='w+', dtype = roi_img.dtype, shape=(n_frames,)+roi_img.shape)
                            arr[f_id] = roi_img
                            arr.flush()
                        else:
                            arr = open_memmap(fn, mode='r+')
                            try:
                                arr[f_id] = roi_img
                                arr.flush()
                                #arr[f_id] = np.append(arr[f_id], np.expand_dims(roi_img,0).copy(), axis=0)
                            except ValueError: #Edge case: ROI fetching has failed giving strange shaped ROI, just leave the zeros as is.
                                pass
                                # roi_stack = np.append(roi_stack, np.expand_dims(np.zeros_like(roi_stack[0]), 0), axis=0)
                        #np.save(self.config['image_path']+os.sep+'spot_images_dir'+os.sep+rn+'.npy', roi_stack)
                        #print(roi_array)
                        #roi_array_padded.append(ip.pad_to_shape(roi, shape = roi_image_size, mode = 'minimum'))
                f_id += 1
        if self.array_id is None: #Only do this if not running distributed, otherwise need to collect and zip later.
            image_io.zip_folder(self.image_handler.image_save_path+os.sep+'spot_images_dir', 'spot_images.npz', remove_folder = True)
            self.image_handler.images['spot_images'] = image_io.NPZ_wrapper(self.image_handler.image_save_path+os.sep+'spot_images.npz')

            
            #for j, pos_roi in enumerate(pos_rois):
            #    roi_array[str(pos)+'_'+str(j).zfill(5)] = pos_roi.copy()
        
        #print(roi_array.keys())
        
        # pos_rois = {}
      
        # for roi_id in rois.roi_id.unique():
        #     try:
        #         pos_rois[str(roi_id).zfill(5)] = np.stack([roi_array[(roi_id, frame)] for frame in range(T)])
        #     except KeyError:
        #         break
        #     except ValueError: #Edge case handling for rois very close to the edge, sometimes the intial padding does not work properly due to rounding errors.
        #         roi_size = roi_array[(roi_id, T-1)].shape
        #         pos_rois[str(roi_id).zfill(5)] = np.stack([ip.pad_to_shape(roi_array[(roi_id, frame)], roi_size) for frame in range(T)])
        
        #self.temp_array = pos_rois
        #roi_array_padded = np.stack(roi_array_padded)

        #print('ROIs generated, saving...')
        #self.image_handler.images['spot_images'] = pos_rois
        #self.image_handler.spot_images['spot_images_padded'] = roi_array_padded
            
        #np.savez(self.config['image_path']+os.sep+'spot_images'+self.postfix+'.npz', **roi_array)
        #self.image_handler.images['spot_images'] = image_io.NPZ_wrapper(self.config['image_path']+os.sep+'spot_images'+self.postfix+'.npz')
        #print('ROIs saved.')
        #np.save(self.image_handler.spot_images_path+os.sep+'spot_images_padded.npy', roi_array_padded)

    def gen_roi_imgs_inmem_coursedc(self):
        # Use this simplified function if the images that the spots are gathered from are already coursely drift corrected!
        #rois = self.roi_table#.iloc[0:500]
        #imgs = self.
        print('Generating single spot image stacks from coursely drift corrected images.')
        rois = self.image_handler.tables[self.config['spot_input_name']+'_dc_rois']
        for pos, group in tqdm.tqdm(rois.groupby('position')):
            pos_index = self.image_handler.image_lists[self.config['spot_input_name']].index(pos)
            full_image = np.array(self.image_handler.images[self.config['spot_input_name']][pos_index])
            #print(full_image.shape)
            for roi in group.to_dict('records'):
                spot_stack = full_image[:, 
                                roi['ch'], 
                                roi['z_min']:roi['z_max'], 
                                roi['y_min']:roi['y_max'],
                                roi['x_min']:roi['x_max']].copy()
                #print(spot_stack.shape)
                fn = pos+'_'+str(roi['frame'])+'_'+str(roi['roi_id_pos']).zfill(4)
                np.save(self.image_handler.image_save_path+os.sep+'spot_images_dir'+os.sep+fn+'.npy', spot_stack)
        #self.image_handler.images['spot_images'] = all_spots
        if self.array_id is None: #Only do this if not running distributed, otherwise need to collect and zip later.
            image_io.zip_folder(self.image_handler.image_save_path+os.sep+'spot_images_dir', 'spot_images.npz', remove_folder = True)
            self.image_handler.images['spot_images'] = image_io.NPZ_wrapper(self.image_handler.image_save_path+os.sep+'spot_images.npz')
        #np.savez_compressed(self.config['image_path']+os.sep+'spot_images.npz', **all_spots)
        #self.image_handler.images['spot_images'] = image_io.NPZ_wrapper(self.config['image_path']+os.sep+'spot_images.npz')
        

    def fine_dc_single_roi_img(self, roi_img, roi):
        #Shift a single image according to precalculated drifts.
        dz = float(roi['z_px_fine'])
        dy = float(roi['y_px_fine'])
        dx = float(roi['x_px_fine'])
        #roi_image_shifted = delayed(ndi.shift)(roi_image_exp, (dz, dy, dx))
        roi_img = ndi.shift(roi_img, (dz, dy, dx)).astype(np.uint16)
        return roi_img

    def gen_fine_dc_roi_imgs(self):
        #Apply fine scale drift correction to spot images, used mainly for visualizing fits (these images are not used for fitting)
        print('Making fine drift-corrected spot images.')

        imgs = self.image_handler.images['spot_images']
        
        rois = self.all_rois

        i = 0
        roi_array_fine = []
        for j, frame_stack in tqdm.tqdm(enumerate(imgs)):
            roi_stack_fine = []
            for roi_stack in frame_stack:
                roi_stack_fine.append(self.fine_dc_single_roi_img(roi_stack, rois.iloc[i]))
                i += 1
            roi_array_fine.append(np.stack(roi_stack_fine))

        #roi_imgs_fine = Parallel(n_jobs=-1, verbose=1, prefer='threads')(delayed(self.fine_dc_single_roi_img)(roi_imgs[i], rois.iloc[i]) for i in tqdm(range(roi_imgs.shape[0])))
        #roi_imgs_fine = np.stack(roi_array_fine)
        #roi_array_fine = np.array(roi_array_fine, dtype='object')
        
        self.image_handler.images['spot_images_fine'] = roi_array_fine
        np.savez_compressed(self.image_handler.image_save_path+os.sep+'spot_images_fine.npz', *roi_array_fine)