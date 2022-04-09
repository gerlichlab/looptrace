def load_spot_images(self):
    '''
    Function to load existing nuclear images from nucs folder in analysis folder.
    '''
    self.spot_images = {}

    if os.path.isdir(self.spot_images_path):
        try:
            image_folders = sorted([(p.name, p.path) for p in os.scandir(self.spot_images_path)])
            for name, path in image_folders:
                name = os.path.splitext(name)[0]
                try:
                    self.spot_images[name] = np.load(path, mmap_mode='r')
                except ValueError:
                    self.spot_images[name] = np.load(path, allow_pickle=True)
                print('Loaded spot images: ', name)
                
        except FileNotFoundError:
            print('Could not find any spot images.')
    else:
        os.makedirs(self.spot_images_path)


def images_from_zarr(self):
    in_path = self.config['input_path']
    filetype = self.config['image_filetype']

    if filetype == 'ome-zarr':
        if os.path.isdir(self.images_path):
            self.images, self.pos_list = ip.multi_ome_zarr_to_dask(self.images_path)
            print(f'Images loaded from zarr store: ', self.images)
        else:
            print('No seq images found.')

    elif filetype == 'zip':
        store = zarr.ZipStore(in_path, mode='r')
        self.images = da.from_zarr(zarr.Array(store))
        #self.images = [imgs[i] for i in range(imgs.shape[0])]
        print(f'Images loaded from zarr store: ', self.images)

    elif filetype == 'zarr':
        self.images = da.from_zarr(in_path)
        #self.images = [imgs[i] for i in range(imgs.shape[0])]
        print(f'Images loaded from zarr store: ', self.images)
    
    elif filetype == 'nikon_tiff':
        self.images = ip.nikon_tiff_to_dask(in_path)
        print(f'Images loaded from tiff folder: ', self.images)

    elif filetype == 'nikon_tiff_multifolder':
        images = []
        for path in sorted(os.listdir(in_path)):
            try:
                print('Reading TIFF files from ', in_path+os.sep+path)
                images.append(ip.nikon_tiff_to_dask(in_path+os.sep+path))
            except ValueError:
                continue
        self.images = da.concatenate(images, axis = 1)
        print(f'Images loaded from multiple tiff folders: ', self.images)
    try:
        self.images = self.images.astype(np.uint16)
    except AttributeError:
        pass


def input_parser(self):
    ft = self.config['image_filetype']
    if ft in ['czi']:
        self.images, self.pos_list, self.all_image_files = ip.images_to_dask(self.config['input_path'], self.config['image_filetype']+self.config['image_template'])
    
    elif ft in ['zip', 'zarr']:
        try:
            self.pos_list = pd.read_csv(self.config['input_path']+os.sep+self.config['output_prefix']+'positions.txt', sep='\n', header=None)[0].to_list()
            print('Position list found: ', self.pos_list)
            self.images_from_zarr()
        except FileNotFoundError:
            self.images_from_zarr()
            self.pos_list = ['P'+str(i).zfill(4) for i in range(1,self.images.shape[0]+1)]
    elif ft == 'ome-zarr':
        self.pos_list = sorted([p.name for p in os.scandir(self.config['input_path']) if os.path.isdir(p)])
        self.images_from_zarr()

    elif ft in ['nikon_tiff', 'nikon_tiff_multifolder']:
        self.images_from_zarr()
        self.pos_list = ['P'+str(i).zfill(4)+'.zarr' for i in range(1,self.images.shape[0]+1)]
    
    else:
        print('Unknown file format, please check config file.')

    try:
        crop = self.config['crop_xy']
        Y = self.images.shape[-2]
        X = self.images.shape[-1]
        newY = int(Y  * (1 - crop) / 2)
        newX = int(X  * (1 - crop) / 2)
        self.images = self.images[..., newY:(Y-newY), newX:(X-newX)]
        print('Images cropped to ', self.images.shape)
    except KeyError: #Crop attribute not set.
        pass

def save_data(self, traces=None, imgs=None, rois=None, pwds=None, pairs=None, config=None, suffix=''):
    '''
    Helper function to save output data from the various modules into the output folder.
    '''
    
    output_path=self.config['output_path']
    output_filename=self.config['output_prefix']
    output_file=output_path+os.sep+output_filename
    
    if traces is not None:
        traces.to_csv(output_file+'traces'+suffix+'.csv', index=None)
    if pwds is not None:
        np.save(output_file+'pwds.npy',pwds)
    if rois is not None:
        rois.reset_index(drop=True, inplace=True)
        rois.to_csv(output_file+'rois'+suffix+'.csv')
    if imgs is not None:
        #imgs=np.moveaxis(imgs,0,2)
        print(imgs.shape)
        tifffile.imsave(output_file+'spot_imgs'+suffix+'.tif', imgs, imagej=True)
    if pairs is not None:
        pairs.to_csv(output_file+'pairs.csv')
    if config is not None:
        with open(output_file+'config.yaml', 'w') as myfile:
            yaml.safe_dump(config, myfile)
    
    print('Data saved')

    ## Legacy classification with ilastic.
    # def classify_nuclei(self):
    #     '''
    #     Runs nucleus classification after detection usign pre-trained ilastik model.
    #     Saves classified images.
    #     '''

    #     print('Running classification of nuclei with Ilastik.')
    #     raw_imgs = [str(p) for p in Path(self.nuc_folder).glob('nuc_raw_*.tiff')] #' '.join(
    #     seg_imgs = [str(p) for p in Path(self.nuc_folder).glob('nuc_binary_*.tiff')]
        
    #     ilastik_path = self.config['ilastik_path']
    #     project_path = self.config['ilastik_project_path']
    #     params = f' --headless --project=\"{project_path}\" --export_source=\"Object Predictions\" --output_format=numpy '
    #     for raw_img, seg_img in zip(raw_imgs, seg_imgs):
    #         raw_data = f'--raw_data {raw_img} '
    #         segmentation = f'--segmentation_image {seg_img}'
    #         command = ilastik_path+params+raw_data+segmentation
    #         subprocess.run(command)
    #     nuc_class = [np.load(img) for img in Path(self.nuc_folder).glob('nuc_raw_*_Object*.npy')]
    #     print('Nucleus classification done.')
    #     self.image_handler.nuc_class = nuc_class

        # def save_nucs(self, img_type):
    #     '''
    #     Function to save nuclear images, either raw or the masks, as tiff files in nucs folder.

    #     Args:
    #         img_type ([str]): Type of images to save, can be 'raw', 'mask' or 'class'.
    #     '''
    #     Path(self.nuc_folder).mkdir(parents=True, exist_ok=True)
    #     imgs = []
    #     for pos in self.pos_list:
    #         pos_index = self.pos_list.index(pos)
    #         if img_type=='raw':
    #             img = self.nucs[pos_index]
    #             tifffile.imsave(self.nuc_folder+os.sep+'nuc_raw_'+pos+'.tiff', data=img)
    #         elif img_type=='mask':
    #             img = self.nuc_masks[pos_index]
    #             tifffile.imsave(self.nuc_folder+os.sep+'nuc_labels_'+pos+'.tiff', data=img)
    #         elif img_type=='class':
    #             img = self.nuc_class[pos_index]
    #             np.save(self.nuc_folder+os.sep+'nuc_raw_'+pos+'_Object Predictions.npy', img)

"""
    def trace_single_frame(self, trace_images, roi, decon_params):
        #trace_ch = self.config['trace_ch']
        roi_image_size = tuple(self.config['roi_image_size'])
        roi_slice = roi['roi_slice']
        pad = roi['pad']

        try:
            roi_image = np.array(trace_images[roi_slice[0], 
                                            roi_slice[1],
                                            roi_slice[2]])

            #If microscope drifted, ROI could be outside image. Correct for this:
            if pad != ((0,0),(0,0),(0,0)):
                #print('Padding ', pad)
                roi_image = np.pad(roi_image, pad, mode='edge')

        except ValueError: # ROI collection failed for some reason
            roi_image = np.zeros(roi_image_size, dtype=np.float32)
        
        if decon_params[0] != 0:
            roi_image = ip.decon_RL(roi_image, decon_params[2], decon_params[1], decon_params[3], niter=decon_params[0])

        #Perform 3D gaussian fit
        #trace_res.append(delayed(self.fit_func)(roi_image, sigma=1, center='max')[0])
        fit = self.fit_func(roi_image, sigma=1, center='max')[0]
        #Ensure images are compatible size for hyperstack.
        
        #roi_image_exp = delayed(ip.pad_to_shape)(roi_image, roi_image_size)
        if roi_image.shape != roi_image_size:
            #print(roi_image.shape)
            roi_image = ip.pad_to_shape(roi_image, roi_image_size)

        #Extract fine drift from drift table and shift image for display.
        dz = float(roi['z_px_fine'])
        dy = float(roi['y_px_fine'])
        dx = float(roi['x_px_fine'])
        #roi_image_shifted = delayed(ndi.shift)(roi_image_exp, (dz, dy, dx))
        roi_image = ndi.shift(roi_image, (dz, dy, dx))

        return fit, roi[['roi_id', 'frame', 'ref_frame', 'position', 'z_px_fine', 'y_px_fine', 'x_px_fine']].values, roi_image

    def tracing_3d(self):
        
        Fits 3D gaussian to previously detected ROIs in all timeframes.
       
        Returns
        -------
        res : Pandas dataframe containing trace data
        imgs : Hyperstack image with raw image data of each ROI.
    

        #Extract parameters from config and predefined roi table.
        decon_params = []
        decon_params.append(self.config['deconvolve'])
        if decon_params[0] != 0:
            algo, kernel, fd_data = ip.decon_RL_setup(  res_lateral=self.config['xy_nm']/1000, 
                                                        res_axial=self.config['z_nm']/1000, 
                                                        wavelength=self.config['spot_wavelength']/1000,
                                                        na=self.config['objective_na'])
            decon_params.append(algo)
            decon_params.append(kernel)
            decon_params.append(fd_data)
        #roi_image_size = self.config['roi_image_size']
        #roi_table = self.roi_table[self.roi_table['position'].isin(self.pos_list)]

        fits = []
        fit_rois = []
        roi_imgs = []


        for pos in sorted(list(self.all_rois['position'].unique())):
            pos_index = self.pos_list.index(pos)
            for t in sorted(list(self.all_rois['frame'].unique())):
                for ch in sorted(list(self.all_rois['ch'].unique())):

                    #print('Loading all images in position ', pos)
                    trace_images = np.array(self.images[pos_index, t, ch]) #self.images[pos_index, t, ch] #
                    sel_rois = self.all_rois[(self.all_rois['position'] == pos) & (self.all_rois['ch'] == ch) & (self.all_rois['frame'] == t)]
                    print(f'Tracing {len(sel_rois)} ROIs in position', pos, ', frame ', t, ', channel ', ch)
                    out = Parallel(n_jobs=-1, prefer='threads')(delayed(self.trace_single_frame)(trace_images, roi, decon_params) for i, roi in tqdm(sel_rois.iterrows(), total=sel_rois.shape[0]))
                    for sublist in out:
                        fits.append(sublist[0])
                        fit_rois.append(sublist[1])
                        roi_imgs.append(sublist[2])
        

        for i, roi in tqdm(self.all_rois.iterrows(), total=self.all_rois.shape[0]):

            #Extract position and ROI coordinates, and extract roi image from 
            #slicable image object (typically a dask array)
            
            roi_slice = roi['roi_slice']
            pad = roi['pad']

            try:
                roi_image = np.array(self.images[roi['pos_index'],
                                                roi['frame'], 
                                                trace_ch,
                                                roi_slice[0], 
                                                roi_slice[1],
                                                roi_slice[2]])

                #If microscope drifted, ROI could be outside image. Correct for this:
                if pad != ((0,0),(0,0),(0,0)):
                    #print('Padding ', pad)
                    roi_image = np.pad(roi_image, pad, mode='edge')

            except ValueError: # ROI collection failed for some reason
                roi_image = np.zeros(roi_image_size, dtype=np.float32)
            
            if decon != 0:
                roi_image = ip.decon_RL(roi_image, kernel, algo, fd_data, niter=decon)

            #Perform 3D gaussian fit
            #trace_res.append(delayed(self.fit_func)(roi_image, sigma=1, center='max')[0])
            trace_res.append(self.fit_func(roi_image, sigma=1, center='max')[0])
            #Ensure images are compatible size for hyperstack.
            
            #roi_image_exp = delayed(ip.pad_to_shape)(roi_image, roi_image_size)
            if roi_image.shape != roi_image_size:
                print(roi_image.shape)
                roi_image = ip.pad_to_shape(roi_image, roi_image_size)

            #Extract fine drift from drift table and shift image for display.
            dz = float(roi['z_px_fine'])
            dy = float(roi['y_px_fine'])
            dx = float(roi['x_px_fine'])
            #roi_image_shifted = delayed(ndi.shift)(roi_image_exp, (dz, dy, dx))
            roi_image = ndi.shift(roi_image, (dz, dy, dx))
            all_images.append(roi_image)
            
            #Add some parameters for tracing table
            trace_index.append([roi['roi_id'], roi['frame'], roi['ref_frame'], roi['position'], dz, dy, dx])

            #Add all the results per timepoint, compute on delayed dask objects.
        #trace_res = dask.compute(*trace_res)
        #all_images = np.stack(dask.compute(*all_images))
        


        #Cleanup of results into dataframe format
        trace_res = pd.DataFrame(fits,columns=["BG","A","z_px","y_px","x_px","sigma_z","sigma_xy"])
        trace_index = pd.DataFrame(fit_rois, columns=["trace_id", "frame", "ref_frame", "position", "drift_z", "drift_y", "drift_x"])
        traces = pd.concat([trace_index, trace_res], axis=1)

        #Apply fine scale drift to fits, and physcial units.
        traces['z_px']=traces['z_px']+traces['drift_z']
        traces['y_px']=traces['y_px']+traces['drift_y']
        traces['x_px']=traces['x_px']+traces['drift_x']
        traces=traces.drop(columns=['drift_z', 'drift_y', 'drift_x'])
        traces['z']=traces['z_px']*self.config['z_nm']
        traces['y']=traces['y_px']*self.config['xy_nm']
        traces['x']=traces['x_px']*self.config['xy_nm']
        traces['sigma_z']=traces['sigma_z']*self.config['z_nm']
        traces['sigma_xy']=traces['sigma_xy']*self.config['xy_nm']
        traces = traces.sort_values(['trace_id', 'frame'])

        #Make final hyperstack of images in PTZYX order.
        #roi_image_size = tuple(self.config['roi_image_size'])
        #all_images = np.reshape(np.stack(all_images), 
        #                        (max(traces.trace_id)+1,max(traces.frame)+1, 
        #                        roi_image_size[0], roi_image_size[1], roi_image_size[2]))
        
        T = self.images_shape[1]
        all_images = np.stack(roi_imgs)
        all_images = np.reshape(all_images, (all_images.shape[0]//T, T, all_images.shape[1], all_images.shape[2], all_images.shape[3])).astype(np.uint16)
        
        if self.trace_beads: 
            suffix = '_beads'
        else:
            suffix = ''
        self.image_handler.save_data(traces=traces, suffix=suffix)
        self.image_handler.save_data(imgs=all_images, suffix=suffix)
        return traces, all_images
""" 
