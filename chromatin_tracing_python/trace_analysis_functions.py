# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 09:19:04 2020

@author: ellenberg
"""
import itertools
import pandas as pd
import numpy as np
from plotly.offline import iplot, plot
import plotly.graph_objs as go
import plotly.express as px
from scipy import interpolate
from scipy import stats
import scipy.ndimage as ndi
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

import matplotlib.pyplot as plt
import napari
from joblib import Parallel, delayed
import dask.array as da
from chromatin_tracing_python import image_processing_functions as ip
import h5py

def tracing_qc(row, qc_dict):
    '''
    Function to set QC value of each fit based on 
    settings from config file.
    Used with pandas.DataFrame.apply()
    '''
    
    A_to_BG = qc_dict['A_to_BG']
    sigma_xy_max = qc_dict['sigma_xy_max']
    sigma_z_max = qc_dict['sigma_z_max']

    if row['A']<(A_to_BG*row['BG']):
        return 0
    elif row['sigma_xy'] > sigma_xy_max or row['sigma_z'] > sigma_z_max:
        return 0
    elif row['sigma_xy'] < 0 or row['sigma_z'] < 0:
        return 0
    elif row['x_px']<0 or row['y_px'] < 0 or row['z_px']<0:
        return 0
    elif row['x_px']>100 or row['y_px'] > 100 or row['z_px'] > 100:
        return 0
    else:
        return 1
    
def group_mean_qc(row, groups):
    '''
    Function to set QC value of each row
    based on group calculation, in this case 
    number of nm away from group mean each point can be.
    Preserves original QC, can only change 1 to 0.
    '''
    #print(groups.iloc[row.name]['z'])
    #min_groups=groups-self.config['max_dist_qc']
    #max_groups=groups+self.config['max_dist_qc']
    max_dist = self.config['max_dist_qc']
    z_mean=groups.iloc[row.name]['z']
    y_mean=groups.iloc[row.name]['y']
    x_mean=groups.iloc[row.name]['x']

    if row['z']>(z_mean+max_dist) or row['z']<(z_mean-max_dist):
        return 0
    if row['y']>(y_mean+max_dist) or row['y']<(y_mean-max_dist):
        return 0
    if row['x']>(x_mean+max_dist) or row['x']<(x_mean-max_dist):
        return 0
    if row['QC'] == 0:
        return 0
    else:
        return 1

def view_context(all_images,
                 contrast= ((0,5000),(0,2000),(0,5000)),
                 trace_id = None, 
                 ref_slice = None,
                 rois = None):
    '''
    Convenvience function to view a given ROI in context in napari.
    If not trace_ID or ROIs given the whole image stack divided by channel is shown.
    '''
    
    colors = ['magenta', 'green', 'blue', 'gray']
    with napari.gui_qt():
        viewer = napari.Viewer()
        if trace_id is not None:
            point = np.array(rois.iloc[trace_id][['zc', 'yc', 'xc']])
            positions = list(rois['position'].unique())
            pos_index = int(positions.index(rois.iloc[trace_id]['position'])) #Get pos_index from W00XX format

            for ch in range(all_images.shape[2]):
                viewer.add_image(all_images[pos_index,:,ch],
                                 contrast_limits=contrast[ch], 
                                 blending='additive', 
                                 colormap=colors[ch])

            point_layer = viewer.add_points(point,
                                            size=8,
                                            edge_width=3,
                                            edge_color='red',
                                            face_color='transparent',
                                            n_dimensional=True)
            sel_dim = np.concatenate([[int(ref_slice)], point.astype(int)])

            for dim in range(len(sel_dim)):
                viewer.dims.set_current_step(dim, sel_dim[dim])
    return viewer

def view_fits(traces, imgs, rois, config, mode='2D', contrast=(100,10000), axis=2):
    '''
    Convenience function to view 3d guassian fits on top of 2D (max z-projection) or 3D spot data.
    '''

    points = points_for_overlay(traces, rois, config)
    if mode == '2D':
        imgs=np.max(imgs, axis=axis)
        with napari.gui_qt():
            viewer = napari.view_image(imgs, contrast_limits=contrast)
            viewer.add_points(points[:,(0,1,3,4)], size=[0,0,1,1], face_color='blue', symbol='cross', n_dimensional=True)
    elif mode == '3D':
        with napari.gui_qt():
            viewer = napari.view_image(imgs, contrast_limits=contrast)
            viewer.add_points(points[:,(0,1,2,3,4)], size=[0,0,3,1,1], face_color='blue', symbol='cross', n_dimensional=True)

def points_for_overlay(traces, rois, config):
    '''
    Generate the fits in a format convenient to display as a marker in napari.
    '''

    roi_image_size = config['roi_image_size']
    points_df = traces.copy()
    for i, roi in rois.iterrows():
        #transp_z=(roi_image_size[0]-(roi['z_max']-roi['z_min']))//2
        #transp_y=(roi_image_size[1]-(roi['y_max']-roi['y_min']))//2
        #transp_x=(roi_image_size[2]-(roi['x_max']-roi['x_min']))//2
        idx = traces['trace_ID'] == roi.name
        points_df[idx] = points_df[idx].assign(z_px = traces[idx]['z_px'],
                                               y_px = traces[idx]['y_px'],
                                               x_px = traces[idx]['x_px'])

    #points_df[['y_px','x_px']]=points_df[['y_px','x_px']].clip(lower=0, upper=64)
    #points_df[['z_px']]=points_df[['z_px']].clip(lower=0, upper=16)
    points=points_df[['trace_ID', 'frame', 'z_px', 'y_px', 'x_px']].to_numpy()
    return points

def eucledian_resolution(traces, frame_names):
    df_sel = traces[traces['frame_name'].isin(frame_names)]
    df_qc = df_sel.groupby(['trace_ID'])[['QC']].sum()
    df_sel = df_sel.groupby(['trace_ID'])[['z','y','x']].diff().dropna().reset_index(drop=True)
    df_sel['eucledian'] = ((0.5*df_sel['z'])**2 + df_sel['y']**2 + df_sel['x']**2)**0.5
    df_sel['QC'] = df_qc['QC']
    df_sel = df_sel[df_sel['QC'] == 2]
    df_sel['id'] = str(frame_names)
    return df_sel

def pwd_calc(traces):
    '''
    Parameters
    ----------
    traces : pd DataFrame with trace data.

    Returns
    -------
    pwds : Pair-wise distance matrixes for traces as an 3-dim numpy array.
    '''
    
    points = [points_from_df_nan(df)[0] for _, df in traces.groupby('trace_ID')]
    pwds = [cdist(p, p) for p in points]
    pwds = np.stack(pwds)
    return pwds

def tracing_length_qc(traces, min_length=0):
    '''
    Parameters
    ----------
    traces : pd DataFrame with trace data.
    pwds : 3dim np array from pwd_calc.
    min_length : Int, minimum length of trace to pass QC

    Returns
    -------
    traces : pd DataFrame with shorter traces removed.
    pwds : 3-dim np array of pwds that passed length QC.

    '''
    grouped=traces.groupby('trace_ID')
    traces_long=grouped.filter(lambda x : x['QC'].sum()>=min_length)
    return traces_long

def trace_analysis(traces, pwds):
    '''
    Calculates pairwise trace similarity based on :
        - MSE of points after rigid alignment
        - MSE of pairwise distance matrices
        - Pearson's correlation coeff of pwds.
        
    Parameters
    ----------
    traces : pd DataFrame with trace data.
    pwds : 3dim np array from pwd_calc.

    Returns
    -------
    output : pd DataFrame with results, including indexes and point 
                coordinates of original and aligned traces.
    '''
    
    pairwise_trace_idx = list(itertools.combinations(traces['trace_ID'].unique(),2))
    pairwise_pwd_idx = list(itertools.combinations(range(pwds.shape[0]),2))
    res = Parallel(n_jobs=-2)(delayed(single_trace_analysis)
                              (traces, pwds, idx1, idx2, idx_p1, idx_p2) for 
                              ((idx1, idx2), (idx_p1, idx_p2)) in 
                              zip(pairwise_trace_idx,pairwise_pwd_idx))

    columns=['idx1', 'idx2', 'aligned_mse', 'aligned_pcc', 'pwd_mse', 'pwd_pcc']
    output=pd.DataFrame(res,columns=columns)
    return output

def single_trace_analysis(traces, pwds, idx1, idx2, idx_p1, idx_p2):
    '''
    Perform pairwise analysis of two single traces according to MSE and PCC metrics
    of their aligned points and their distance matrices.
    '''

    #Get points by their trace indices.
    points_A, points_B = points_from_traces(traces, [idx1,idx2])
    #Align the point sets, note rigid_transform include matching.
    aligned_points = rigid_transform_3D(points_B, points_A)
    #Need to match seperately for the mat_corr functions
    points_A, points_B = match_two_pointsets(points_A, points_B)
    aligned_mse = mat_corr_rmse(points_A, points_B)
    aligned_pcc = mat_corr_pcc(points_A, points_B)
    
    pwd_mse = mat_corr_rmse(pwds[idx_p1],pwds[idx_p2])
    pwd_pcc = mat_corr_pcc(pwds[idx_p1],pwds[idx_p2])

    return idx1, idx2, aligned_mse, aligned_pcc, pwd_mse, pwd_pcc
    
def trace_clustering(paired, metric='pwd_pcc', method='single', color_threshold=None):
    '''
    Calculates clusters nased on similarity metrics of 
    pairwise analysis of traces. Also plots dendrogram of clusters.
    
    Parameters
    ----------
    paired : Paired analysis DataFrame, output of trace_analysis
    metric : One of the similarity metrics from trace_analysis:
            - 'aligned_mse'
            - 'aligned_pcc'
            - 'pwd_mse'
            - 'pwd_pcc'
    method : see scipy.cluster.hierarchy.linkage documentation
    Returns
    -------
    Z : Clustering matrix based on hierarchial clustering (see scipy.cluster.hierarchy.single docs)
    '''
    
    labels=list(np.unique(np.concatenate((paired['idx1'],paired['idx2']))))
    if metric == 'aligned_mse' or metric == 'pwd_mse':
        Z=linkage(paired[metric], method=method)
    elif metric == 'aligned_pcc' or metric == 'pwd_pcc':
        Z=linkage(1-paired[metric], method=method)
    else:
        raise ValueError('Inappropriate metric.')
    plt.figure(figsize=(20,20))
    dendro=dendrogram(Z, labels=labels, color_threshold=color_threshold, leaf_font_size=12)
    if color_threshold is None:
        color_threshold = 0.7*max(Z[:,2])
    clusters=fcluster(Z, color_threshold, criterion='distance')
    cluster_df=pd.DataFrame([labels, clusters]).T 
    cluster_df.columns=['trace_ID', 'cluster']
    
    return cluster_df

    '''
    # Get points from input dataframe.


    A_orig, A_orig_idx = points_from_df(df_A)
    B_orig, B_orig_idx = points_from_df(df_B)

    # Ensure only matching points are used.
    # TODO: This is quite slow, bulk of this whole function. Implementing something faster should be easy.
    A, idx_A, B, idx_B = matching_points_from_dfs(df_A,df_B)
    '''

def run_gpa_all_clusters(traces, cluster_df, min_cluster = 1):
    '''
    Running function to perform GPA analysis on all clusters identified in trace_clustering()
    with number of members above min_cluster.
    '''

    #Find unique cluster IDs from clustering table.
    cluster_ids=set(cluster_df['cluster'])
    #Generate list of lists of all cluster members over min_cluster length.
    all_cluster_members = []
    for cluster_id in cluster_ids:
        cluster_members = list(cluster_df[cluster_df['cluster']==cluster_id]['trace_ID'])
        if len(cluster_members)>=min_cluster:
            all_cluster_members.append(cluster_members)

    #Perform GPA analysis on each of the clusters seperately.
    all_mean_points = [general_procrustes_analysis(traces, cluster_members)[1]
                        for cluster_members in all_cluster_members]
    #Choose random cluster mean as template for alignment.
    template = all_mean_points[np.random.randint(0,len(all_cluster_members))]
    #Align all cluster means to template.
    aligned_mean_points = [rigid_transform_3D(mean_points, template) for 
                            mean_points in all_mean_points]
    #Readd the template to the output.
    #aligned_mean_points += [template]

    return aligned_mean_points

def general_procrustes_analysis(traces, trace_ids, crit=0.01):
    '''
    General procrustes analysis is performed as described in ...
    Runs until procrustes distance is less than crit.

    Returns all the aligned traces, the mean trace and the std of all the aligned traces.
    '''

    trace_ids=list(trace_ids)
    
    # Make list of all points of selected traces
    all_points = points_from_traces(traces, trace_ids)

    # Select a random template for initial loop
    #np.random.seed(1)
    t_idx = np.random.randint(0,len(all_points))
    template = all_points[t_idx]

    #The initial distance before alignment.
    prev_dist = np.sum([procrustes_distance(template, points) for 
                   points in all_points])
    print('Initial distance: ', prev_dist)
    print('Number of traces: ', len(all_points))
    #Run the first alignment step:
    all_points, points_mean, dist = general_procrustes_loop(all_points, template)
    
    #Run the remaining alignment steps until crit is reached:
    n_cycles = 0
    while np.abs(prev_dist-dist) > crit:
        prev_dist = dist
        all_points, points_mean, dist = general_procrustes_loop(all_points, points_mean)
        n_cycles += 1
        
    print('GPA converged after {} cycles with distance {}'.format(n_cycles, dist))
    
    #Calculate standard deviation of all points:
    points_std = np.nanstd(np.stack(all_points), axis = 0)

    return all_points, points_mean, points_std
        
def general_procrustes_loop(all_points, template):
    '''
    A single cycle in the general procrustes analysis.
    Returns the points in all_points aligned to template,
    the mean points and the procrustes distance to the mean.
    '''
    # Align all point sets to mean template
    all_points_aligned = [rigid_transform_3D(offset, template) for 
                          offset in all_points]
    
    #Set values that do not pass QC to nan in new list.
    all_points_aligned_qc=[]
    for points in all_points_aligned:
        points[points[:,3] == 0, 0:3]=np.nan
        all_points_aligned_qc.append(points)
    
    #Calculate mean points from QC list ignoring nans.:
    points_mean = np.nanmean(np.stack(all_points_aligned_qc), axis=0)
    #The "QC" for the mean is 1 if at least one element has a QC=1
    points_mean[:,3]=np.ceil(points_mean[:,3])
    # Calculate distance to mean:
    dist = np.sum([procrustes_distance(points_mean, points) for 
                   points in all_points_aligned])
    return all_points_aligned, points_mean, dist
    
def procrustes_distance(points_A, points_B):
    '''
    Procrustes distance (identical to RMSE) between two point sets.
    '''

    points_A, points_B = match_two_pointsets(points_A, points_B)    
    dist = np.sqrt(np.mean((points_A-points_B)**2))
    return dist
    
def rigid_transform_3D(points_A, points_B):
    '''
    Calculates rigid transformation of two 3d points sets based on:
    Least-squares fitting of two 3-D point sets. IEEE T Pattern Anal 1987
    DOI: 10.1109/TPAMI.1987.4767965
    
    Finds the optimal (lest squares) of B = RA + t, so mapping of A onto B.
    
    Modified from http://nghiaho.com/?page_id=671
    
    Only uses points present in both traces for alignment.

    Parameters
    ----------
    points_A, points_B : Nx4 (ZYX + condition) numpy ndarrays.

    Returns
    -------
    Coordinates of registered and transformed points_A
    '''
    
    #Ensure matching points
    #Need column vectors for calculation
    A, B = match_two_pointsets(points_A, points_B)

    A = A.T
    B = B.T
    
    # Check that we have 3D column vectors
    num_rows, num_cols = A.shape

    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

    [num_rows, num_cols] = B.shape
    if num_rows != 3:
        raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

    # find mean column wise and reshape to column vector (centroids)
    Ac = np.mean(A, axis=1).reshape((3,1))
    Bc = np.mean(B, axis=1).reshape((3,1))

    # subtract mean
    Am = A - Ac
    Bm = B - Bc
    
    # Calculate covariance matrix
    H = np.matmul(Am, Bm.T)
    
    # Find optimal rotation by SVD of the covariance matrix.
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T,U.T)

    # Handle case if the rotation matrix is reflected.
    if np.linalg.det(R) < 0:
        #print("det(R) < R, reflection detected!, correcting for it ...\n");
        Vt[2,:] *= -1
        R = np.matmul(Vt.T,U.T)

    # calculate translation.
    t = Bc - np.matmul(R,Ac)
    
    # Transform the matched points using calculated R and t
    #A_reg = np.matmul(R,A)+t
    
    #Calculate the RMSE of the alignment
    # rmse = np.sqrt(np.mean(np.sum((A_reg-B)**2)))
    
    #Transform the original vector with QC values
    points_A_reg = np.copy(points_A)
    points_A_reg[:,:3] = (np.matmul(R, points_A[:,:3].T)+t).T
    
    return points_A_reg#, rmse

'''
def scale_rigid_transform_3D(df_A,df_B):
    A_orig, A_orig_idx = points_from_df(df_A)
    B_orig, B_orig_idx = points_from_df(df_B)
    A, idx_A, B, idx_B = matching_points_from_dfs(df_A,df_B)
    reg = RigidRegistration(**{'X': A, 'Y': B})
    B_ = reg.transform_point_cloud(B)
    mse = ((A - B_)**2).mean()
    B_aligned = reg.transform_point_cloud(B_orig)
    return B_aligned, B_orig_idx, mse
'''

def match_two_pointsets(points_A, points_B):
    '''
    Matches two point sets by their QC value to only return points
    passing QC in both sets.

    Parameters
    ----------
    points_A, points_B : Nx4 (ZYX + QC) numpy ndarrays.

    Returns
    -------
    points_A_matched, points_B_matched : Nx3 (ZYX) numpy ndarrays.

    '''

    match_idx = points_A[:,3] * points_B[:,3] != 0
    points_A_matched = points_A[match_idx,0:3]
    points_B_matched = points_B[match_idx,0:3]
    return points_A_matched, points_B_matched

def points_from_traces(traces, trace_ids):
    '''
    Helper function to extract point coordinates from trace dataframe.
    Only points passing QC during tracing are returned.

    Parameters
    ----------
    traces : pd DataFrame with trace data.
    trace_ids: single or multiple trace_ids to extract

    Returns
    -------
    points : list of  Nx4 np array with trace coordinates and QC value.
    '''
    
    if not isinstance(trace_ids, (list, tuple)):
        trace_ids = [trace_ids]
    points_with_qc=[]
    for trace_id in trace_ids:
        idx = traces['trace_ID'] == trace_id
        points_with_qc.append(traces.loc[idx,['z','y','x','QC']].to_numpy())
    return points_with_qc

def points_from_df_nan(trace_df):
    '''
    Helper function to extract point coordinates from trace dataframe.
    All points are returned, but points not passing QC during tracing 
    are returned as NaN.   

    Parameters
    ----------
    trace_df : pd DataFrame with trace data.

    Returns
    -------
    points : Nx3 np array with trace coordinates.
    idx : List, Original index of trace coordinates.
    '''
    
    _traces=trace_df.copy()
    _traces[_traces['QC']==0] = np.nan
    points = np.array([_traces['x'].values,_traces['y'].values,_traces['z'].values]).T
    idx = _traces['frame']
    return points, idx

def matching_points_from_dfs(df_A,df_B):
    '''
    Helper function to find points that are common to two traces.

    Parameters
    ----------
    df_A, df_B : pd DataFrames with single trace data.

    Returns
    -------
    Coordinates (np array) and indexes (list) of the matching points.

    '''
    df_A_idx=df_A[df_A['QC']==1]['frame']
    df_B_idx=df_B[df_B['QC']==1]['frame']
    idx=np.intersect1d(df_A_idx, df_B_idx)
    i = pd.IndexSlice
    points_A, idx_A = points_from_df(df_A.loc[df_A['frame'].isin(idx)])
    points_B, idx_B = points_from_df(df_B.loc[df_B['frame'].isin(idx)])
    return points_A, idx_A, points_B, idx_B

def spline_interp(points):
    '''
    Performs cubic B-spline interpolation on point coordinates.

    Parameters
    ----------
    points : n_points X ndim list or array of point coordinates.

    Returns
    -------
    fine : 100 X ndim nd array of interpolated points.

    '''
    
    tck, u = interpolate.splprep(points, s=0)
    knots = interpolate.splev(tck[0], tck)
    u_fine = np.linspace(0,1,num=100)
    fine = interpolate.splev(u_fine, tck)
    return fine

def mat_corr_rmse(mat1, mat2):
    '''
    Calculate mean squared error of two matrices, ignoring nans.
    '''
    rmse = np.sqrt(np.nanmean((mat1-mat2)**2))
    return rmse

def mat_corr_pcc(mat1,mat2):
    '''
    Calculate pearson's corr coef of two matrices, ignoring nans.
    '''
    mat1_bar=np.nanmean(mat1)
    mat2_bar=np.nanmean(mat2)
    
    pcc_num=np.nansum((mat1-mat1_bar)*(mat2-mat2_bar))
    pcc_denom=np.sqrt(np.nansum((mat1-mat1_bar)**2)*np.nansum((mat2-mat2_bar)**2))
    pcc=pcc_num/pcc_denom
    
    return pcc


def radius_of_gyration(point_set):
    '''
    Calculate ROG: R = sqrt(1/N * sum((r_k - r_mean)^2) for k points in structure.) 
    Source: https://en.wikipedia.org/wiki/Radius_of_gyration
    '''

    #Only include points passing QC:
    qc_idx = point_set[:,3] != 0
    point_set_qc = point_set[qc_idx, 0:3]

    points_mean=np.mean(point_set_qc, axis=0)
    rog = np.sqrt(1/points_mean.shape[0] * np.sum((point_set_qc - points_mean)**2))
    return rog

def elongation(point_set):
    '''
    Elongation in this case is defined as the ratio between the two primary
    eigenvalues of the point set.
    '''

    #Only include points passing QC:
    qc_idx = point_set[:,3] != 0
    point_set_qc = point_set[qc_idx, 0:3]

    #Center points to 0-mean
    points_centered = point_set_qc - np.mean(point_set_qc, axis=0)
    n, m = points_centered.shape
    #Compute covariance matrix
    cov = np.dot(points_centered.T, points_centered) / (n-1)
    #Eigenvector decomposition of covariance matrix
    eigen_vals, eigen_vecs = np.linalg.eig(cov)
    #Elongation is the ratio of the secondary eigenvalue to primary eigenvalue
    eigen_vals = np.sort(eigen_vals)[::-1]
    #print('Eigenvalues are ', eigen_vals)
    elongation = 1-(eigen_vals[1]/eigen_vals[0])
    return elongation

def contour_length(point_set):
    #Only include points passing QC:
    qc_idx = point_set[:,3] != 0
    point_set_qc = point_set[qc_idx, 0:3]
    dist = 0
    for i in range(point_set_qc.shape[0]-1):
        dist += np.linalg.norm(point_set_qc[i+1]-point_set_qc[i])
    return dist

def plot_traces(traces, trace_id, split=False):
    '''
    Helper function for plotting one or several traces in one figure.
    Also plots spline interpolation between points for visualization.
    
    Parameters
    ----------
    traces : pd DataFrame with trace data.
    trace_id : Int or list of ints with trace_ID of traces to plot.     

    Returns
    ----------
    Fig object for saving or further manipulation.
    '''

    if type(trace_id) == int:
        trace_id=[trace_id]
    df=traces[traces['trace_ID'].isin(trace_id)]
    df['keys'] = df['frame_name'].astype(str).str[0]
    labels=list(df['frame_name'])
    print(labels)

    fig = px.scatter_3d(df, x='x', y='y', z='z', 
                symbol='keys',
                color='frame_name', 
                color_discrete_sequence = px.colors.sequential.thermal[2:],
                labels={'frame_name' : 'Exchange', 'keys':'Group'})

    for i in trace_id:
        for key in list(df['keys'].unique()):
            df_i = df[(df['trace_ID'] == i) & (df['keys'] == key)]
            #print(df_i)
            z_f, y_f, x_f=spline_interp([df_i['z'].values,
                                        df_i['y'].values,
                                        df_i['x'].values])
            fig.add_trace(go.Scatter3d(x=x_f, 
                                    y=y_f, 
                                    z=z_f,
                                    mode ='lines',
                                    showlegend=False,
                                    line=dict(color=np.random.randint(1),
                                                width=4)))

    fig.update_layout(template='plotly_dark', 
                      margin=dict(l=20, r=20, t=20, b=20),
                      showlegend= True,
                        legend=dict(
                                font=dict(
                                    size=10)))
    iplot(fig)
    
    return fig

def plot_aligned_traces(traces, idx):
    '''
    Helper function for plotting one or several aligned traces in one figure.
    Also plots spline interpolation between points for visualization.
    
    Parameters
    ----------
    traces : pd DataFrame with trace data.
    trace_id : Int or list of ints with trace_ID of traces to plot. 

    Returns
    ----------
    Fig object for saving or further manipulation.

    '''


    all_points = points_from_traces(traces, idx)
    template = all_points.pop(0)
    all_points_aligned = [template]+[rigid_transform_3D(offset, template) for 
                          offset in all_points]
    scatters = []
    cmap = px.colors.sequential.thermal[2:]
    for point_id, point_set in enumerate(all_points_aligned):
        idx=np.arange(point_set.shape[0])
        qc_idx = point_set[:,3] != 0
        idx=idx[qc_idx]
        labels=['E'+str(i) for i in idx]
        cmap_points = [cmap[i%10] for i in idx]
        z,y,x = point_set[qc_idx, 0], point_set[qc_idx, 1], point_set[qc_idx, 2]
        z_f, y_f, x_f=spline_interp([z,y,x])
        scatters.append(go.Scatter3d(x=x, y=y, z=z, 
                                     mode='markers', 
                                     marker_color=cmap_points,
                                     marker_size=9,
                                     opacity=1,
                                     name='Trace '+str(point_id)))
        scatters.append(go.Scatter3d(x=x_f, 
                            y=y_f, 
                            z=z_f,
                            mode ='lines',
                            showlegend=False,
                            line=dict(color=px.colors.qualitative.Plotly[point_id],
                                        width=5)))
    fig = go.Figure(data=scatters)
    fig.update_layout(template='plotly_dark', showlegend= False)
    iplot(fig)
    return fig
    
def plot_gpa_output(aligned_points, mean_points, cluster_members):
    '''
    Helper function for plotting the results of a GPA analysis.
    
    Parameters
    ----------
    aligned_points : List of aligned point sets
    mean_points: Nx4 array of mean points
    cluster_members: List of trace_IDs of the aligned points, typically from trace_clustering output

    Returns
    ----------
    Fig object for saving or further manipulation.

    '''
    
    scatters = []
    cmap = px.colors.sequential.thermal
    for point_id, point_set in enumerate(aligned_points):
        idx=np.arange(point_set.shape[0])
        qc_idx = point_set[:,3] != 0
        idx=idx[qc_idx]
        labels=['E'+str(i) for i in idx]
        
        cmap_points = [cmap[i%12] for i in idx]
        
        z,y,x = point_set[qc_idx, 0], point_set[qc_idx, 1], point_set[qc_idx, 2]

        scatters.append(go.Scatter3d(x=x, y=y, z=z, 
                                     mode='markers', 
                                     marker_color=cmap_points,
                                     marker_size=4,
                                     opacity=0.8,
                                     name='Trace '+str(cluster_members[point_id]),
                                     ))
        scatters.append(go.Scatter3d(x=x,y=y,z=z, 
                                       mode='lines',
                                       line=dict(color='rgba(15, 60, 90, 0.5)',
                                               width=3)))
    
    mean_idx=np.arange(mean_points.shape[0])
    mean_qc = mean_points[:,3] != 0
    mean_idx=mean_idx[mean_qc]
    print(mean_idx)
    #mean_labels=['E'+str(i) for i in mean_idx]
    mean_cmap = [cmap[i%12] for i in mean_idx]
    z_m, y_m, x_m = mean_points[mean_idx, 0], mean_points[mean_idx, 1], mean_points[mean_idx, 2]
    
    mean_fig = scatters.append(go.Scatter3d(x=x_m, y=y_m, z=z_m, 
                                            mode='markers+lines', 
                                            marker_color=mean_cmap,
                                            name='Mean',
                                            line=dict(color='#ff7f0e', 
                                                      width=6)))
    

    fig = go.Figure(data=scatters)
    fig.update_layout(template='plotly_dark', showlegend= True)
    iplot(fig)
    return fig
    
def plot_multi_points(list_of_points, names = None):
    '''
    Helper function for plotting, typically used to plot the results of a GPA analysis 
    for all clusters.
    
    Parameters
    ----------
    list_of_points : List of Nx4 ndarray (zyx+QC) point sets
    names: Optinal list of length equals to list_of_points with names of point sets.

    Returns
    ----------
    Fig object for saving or further manipulation.

    '''
    
    scatters = []
    cmap = px.colors.sequential.thermal
    for point_id, point_set in enumerate(list_of_points):
        idx=np.arange(point_set.shape[0])
        qc_idx = point_set[:,3] != 0
        idx=idx[qc_idx]
        labels=['E'+str(i) for i in idx]
        
        cmap_points = [cmap[i%12] for i in idx]
        if names is not None:
            name = names[point_id]
        else:
            name = 'Cluster '+str(point_id)
        point_set_plot = point_set[qc_idx, 0:3]
        scatters.append(go.Scatter3d(x=point_set_plot[:,2], y=point_set_plot[:,1], z=point_set_plot[:,0], 
                                     mode='markers+lines', 
                                     marker_color=cmap_points,
                                     marker_size=6,
                                     opacity=1,
                                     name=name,
                                     line=dict(color=np.random.randint(1),
                                               width=4)))
    
    fig = go.Figure(data=scatters)
    fig.update_layout(template='plotly_dark', showlegend= True, height=600)
    iplot(fig)
    return fig