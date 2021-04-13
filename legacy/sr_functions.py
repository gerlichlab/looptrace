# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 06:19:23 2020

@author: ellenberg
"""

def render_gauss_const(df,sigma,cam_px,cam_nm,grid_px_size, output_res):
    X = df[['xnm','ynm']].to_numpy()/grid_px_size
    X=np.round(X).astype(int)
    grid=np.zeros((cam_px*cam_nm//grid_px_size,cam_px*cam_nm//grid_px_size))
    point=np.zeros((13,13))
    point[6,6]=1
    point=ndi.gaussian_filter(point,sigma)

    for xval,yval in zip(X[:,0],X[:,1]):
        grid[xval-6:xval+7,yval-6:yval+7]+=point
        
    grid=cv2.resize(grid,(output_res, output_res))
    grid=np.flip(np.rot90(grid),0)
    plt.imshow(np.clip(grid,0,1),cmap='gist_heat')
    return grid

def render_gauss(df,pixel_size):
    X = df[['xnm','ynm']].to_numpy()/pixel_size
    X=np.round(X).astype(int)
    err = df['xnmerr'].to_numpy()/pixel_size
    grid=np.zeros((512*103//pixel_size,512*103//pixel_size))
    point=np.zeros((13,13))
    point[6,6]=1

    for xval,yval,e in zip(X[:,0],X[:,1],err):
        point=ndi.gaussian_filter(point,e)
        grid[xval-6:xval+7,yval-6:yval+7]+=point
        
    grid=cv2.resize(grid,(512,512))
    plt.imshow(np.flip(np.rot90(np.clip(grid,0,0.5)),0),cmap='gist_heat')

def render_hist(df):
    X = df[['xnm','ynm']].to_numpy()/pixel_size
    heatmap, xedges, yedges = np.histogram2d(X[:,0], X[:,1], bins=1000)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    heatmap = cv2.resize(np.flip(heatmap.T,(0)),(512,512))
    plt.imshow(np.clip(heatmap,0,10), extent=extent, origin='lower',cmap='gist_heat')
    
    
def clust_dbscan(df, eps, min_samples, hdb=False):
    X = df[['xnm','ynm']].to_numpy()
    xi, yi = np.mgrid[0:np.max(X[:,0]):512j,0:np.max(X[:,1]):512j]
    # #############################################################################
    # Compute DBSCAN
    print('Total number of localizations: %d' % X.shape[0])
    if hdb:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_samples)
        labels = clusterer.fit_predict(X)
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[labels>-1] = True
        
    else:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    
    
    # #############################################################################
    '''
    Plotting:
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = (labels == k)
    
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=None, markersize=5)
    
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], '.',
                 markeredgecolor='k', markersize=0.05)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    '''
    return core_samples_mask
    
def clust_kde(df):
    X = df[['xnm','ynm']].to_numpy()
    xi, yi = np.mgrid[0:np.max(X[:,0]):512j,0:np.max(X[:,1]):512j]
    xy_i=np.vstack((np.ravel(xi),np.ravel(yi))).T
    kde = KernelDensity(kernel='gaussian', bandwidth=10).fit(X)
    Z = np.exp(kde.score_samples(xy_i))
    plt.contourf(xi,yi,np.clip(Z.reshape(512,512),0,1e-9))
    
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