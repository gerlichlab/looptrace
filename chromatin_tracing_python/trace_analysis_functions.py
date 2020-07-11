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


def plot_fits(traces, imgs, mode='2D', contrast=(100,10000)):
    points = points_for_overlay(traces)
    if mode == '2D':
        imgs=np.max(imgs, axis=2)
        with napari.gui_qt():
            viewer = napari.view_image(imgs, contrast_limits=contrast)
            viewer.add_points(points[:,(0,1,3,4)], size=[0,0,1,1], face_color='blue', symbol='cross', n_dimensional=True)
    elif mode == '3D':
        with napari.gui_qt():
            viewer = napari.view_image(imgs, contrast_limits=contrast)
            viewer.add_points(points[:,(0,1,2,3,4)], size=[0,0,3,1,1], face_color='blue', symbol='cross', n_dimensional=True)

def points_for_overlay(traces):
    points_frame=traces.reset_index()[['trace_ID','frame','z_px','y_px','x_px']]
    points_frame[['y_px','x_px']]=points_frame[['y_px','x_px']].clip(lower=0, upper=64)
    points_frame[['z_px']]=points_frame[['z_px']].clip(lower=0, upper=16)
    points=points_frame.to_numpy()
    return points

def pwd_calc(traces):
    '''
    Parameters
    ----------
    traces : pd DataFrame with trace data.

    Returns
    -------
    pwds : Pair-wise distance matrixes for traces as an 3-dim numpy array.
    '''
    
    points = [points_from_df_nan(df)[0] for _, df in traces.groupby(level=0)]
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
    grouped=traces.groupby(level=0)
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
    
    pairwise_trace_idx = list(itertools.combinations(traces.index.unique(),2))
    pairwise_pwd_idx = list(itertools.combinations(range(pwds.shape[0]),2))
    res = Parallel(n_jobs=-2)(delayed(single_trace_analysis)
                              (traces, pwds, idx1, idx2, idx_p1, idx_p2) for 
                              ((idx1, idx2), (idx_p1, idx_p2)) in 
                              zip(pairwise_trace_idx,pairwise_pwd_idx))

    columns=['idx1', 'idx2', 'aligned_mse', 'aligned_pcc', 'pwd_mse', 'pwd_pcc']
    output=pd.DataFrame(res,columns=columns)
    return output

def single_trace_analysis(traces, pwds, idx1, idx2, idx_p1, idx_p2):
    point_set = points_from_traces(traces, [idx1,idx2])
    aligned_points = rigid_transform_3D(point_set[1], point_set[0])
    
    aligned_mse = mat_corr_rmse(point_set[0][:,:3], aligned_points[:,:3])
    aligned_pcc = mat_corr_pcc(point_set[0][:,:3], aligned_points[:,:3])
    
    pwd_mse = mat_corr_rmse(pwds[idx_p1],pwds[idx_p2])
    pwd_pcc = mat_corr_pcc(pwds[idx_p1],pwds[idx_p2])
    return idx1, idx2, aligned_mse, aligned_pcc, pwd_mse, pwd_pcc
    
def trace_clustering(paired, metric='pwd_pcc', method='single', color_threshold=1.0):
    '''
    Calculates clusters nased on similarity metrics of 
    pairwise analysis of traces. Also plots dendrogram of clusters.
    
    Parameters
    ----------
    paired : Paired analysis DataFrame, output of trace_analysis
    metric : One of the similarity metrics from trace_analysis:
            - 'aligned_mse'
            - 'pwd_mse'
            - 'pwd_pcc'
    method : see scipy.cluster.hierarchy.linkage documentation
    Returns
    -------
    Z : Clustering matrix based on hierarchial clustering (see scipy.cluster.hierarchy.single docs)
    '''
    
    labels=np.unique(np.concatenate((paired['idx1'],paired['idx2'])))
    if metric == 'aligned_mse' or metric == 'pwd_mse':
        Z=linkage(np.log(paired[metric]), method=method)
    elif metric == 'aligned_pcc' or metric == 'pwd_pcc':
        Z=linkage(1-paired[metric], method=method)
    else:
        raise ValueError('Inappropriate metric.')
    plt.figure(figsize=(15,15))
    dendro=dendrogram(Z, labels=labels, color_threshold=color_threshold, leaf_font_size=12)
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
    #Find unique cluster IDs from clustering table.
    cluster_ids=set(cluster_df['cluster'])
    
    #Generate list of lists of all cluster members over min_cluster length.
    all_cluster_members = []
    for cluster_id in cluster_ids:
        cluster_members = list(cluster_df[cluster_df['cluster']==cluster_id]['trace_ID'])
        if len(cluster_members)>=min_cluster:
            all_cluster_members.append(cluster_members)

    print(all_cluster_members)
    #Perform GPA analysis on each of the clusters seperately.
    all_mean_points = [general_procrustes_analysis(traces, cluster_members)[1]
                        for cluster_members in all_cluster_members]
    #Choose the first cluster mean as template for alignment.
    template = all_mean_points.pop(0)
    #Align all other cluster means to template.
    aligned_mean_points = [rigid_transform_3D(mean_points, template) for 
                            mean_points in all_mean_points]
    #Readd the template to the output.
    aligned_mean_points += [template]
    return aligned_mean_points

def general_procrustes_analysis(traces, trace_ids, crit=0.01):
    
    trace_ids=list(trace_ids)
    
    # Make list of all points of selected traces
    all_points = points_from_traces(traces, trace_ids)
    
    # Select a random template for initial loop
    #np.random.seed(1)
    t_idx = np.random.randint(0,len(all_points))
    template = all_points[t_idx]
    prev_dist = np.sum([procrustes_distance(template, points) for 
                   points in all_points])
    print('Initial distance is', prev_dist)
    
    all_points, points_mean, dist = general_procrustes_loop(all_points, template)
    
    n_cycles = 0
    while np.abs(prev_dist-dist) > crit:
        prev_dist = dist
        all_points, points_mean, dist = general_procrustes_loop(all_points, points_mean)
        n_cycles += 1
        
    print('GPA converged after {} cycles with distance {}'.format(n_cycles, dist))
    return all_points, points_mean
        
def general_procrustes_loop(all_points, template):
    # Align all to template
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
    Coordinates of registered and transformed points_B
    '''
    
    #Ensure matching points
    #Need column vectors for calculation
    A, B = match_two_pointsets(points_A, points_B)

    A = A.T
    B = B.T
    
    # Check that we have 3D column vectors
    num_rows, num_cols = A.shape;

    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

    [num_rows, num_cols] = B.shape;
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
        points_with_qc.append(traces.loc[trace_id,['z','y','x','QC']].to_numpy())
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
    #Only include points passing QC:
    qc_idx = point_set[:,3] != 0
    point_set_qc = point_set[qc_idx, 0:3]

    # Calculate ROG: R = sqrt(1/N * sum((r_k - r_mean)^2) for k points in structure.) 
    # Source: https://en.wikipedia.org/wiki/Radius_of_gyration
    points_mean=np.mean(point_set_qc, axis=0)
    rog = np.sqrt(1/points_mean.shape[0] * np.sum((point_set_qc - points_mean)**2))
    return rog

def elongation(point_set):
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
    print('Eigenvalues are ', eigen_vals)
    elongation = 1-(eigen_vals[1]/eigen_vals[0])
    return elongation

def plot_traces(traces, idx):
    '''
    Helper function for plotting one or several traces in one figure.
    Also plots spline interpolation between points for visualization.
    
    Parameters
    ----------
    traces : pd DataFrame with trace data.
    idx : Int or list of ints with trace_ID of traces to plot.     
    '''

    if type(idx) == int:
        idx=[idx]
    trace_df=traces[traces['QC']==1]
    trace_df_sel=pd.concat([trace_df.xs(i) for i in idx])
    trace_index=list(trace_df_sel['frame'])
    #print([i for i in trace_index])
    labels=['E'+str(t) for t in trace_index]
    print(labels)
    fig = px.scatter_3d(trace_df_sel, x='x', y='y', z='z',
              color=labels)
    for i in idx:
        interp=spline_interp([trace_df.xs(i)['z'].values,
                         trace_df.xs(i)['y'].values,
                         trace_df.xs(i)['x'].values])
        fig.add_trace(go.Scatter3d(x=interp[2], 
                                   y=interp[1], 
                                   z=interp[0],
                                  mode ='lines',
                                  showlegend=False))
    iplot(fig)

def plot_aligned_traces(traces, idx):
    all_points = points_from_traces(traces, idx)
    template = all_points.pop(0)
    all_points_aligned = [template]+[rigid_transform_3D(offset, template) for 
                          offset in all_points]
    scatters = []
    cmap = px.colors.qualitative.Plotly
    for point_id, point_set in enumerate(all_points_aligned):
        idx=np.arange(point_set.shape[0])
        qc_idx = point_set[:,3] != 0
        idx=idx[qc_idx]
        labels=['E'+str(i) for i in idx]
        
        cmap_points = [cmap[i%10] for i in idx]
        
        point_set_plot = point_set[qc_idx, 0:3]
        scatters.append(go.Scatter3d(x=point_set_plot[:,2], y=point_set_plot[:,1], z=point_set_plot[:,0], 
                                     mode='markers+lines', 
                                     marker_color=cmap_points,
                                     marker_size=9,
                                     opacity=1,
                                     name='Trace '+str(point_id),
                                     line=dict(color='#1f77b4',
                                               width=1)))

    fig = go.Figure(data=scatters)
    iplot(fig)
    
def plot_paired_traces(traces, idxs):
    '''
    Helper function for plotting two aligned traces in one figure.
    Also plots spline interpolation between points for visualization.
    
    Parameters
    ----------
    pair_df : pd DataFrame with paired trace analysis, output of trace_analysis.
    idx : Int, index of pair from paired dataframe. 
    '''
    
    points1=points_from_traces(traces, trace_ids)
    idx1=pair_df['A_idx'][idx]
    points2=pair_df['B_aligned'][idx]
    idx2=pair_df['B_aligned_idx'][idx]

    z1,y1,x1=points1.T
    z2,y2,x2=points2.T
    
    z_fine1,y_fine1,x_fine1=spline_interp([z1,y1,x1])
    z_fine2,y_fine2,x_fine2=spline_interp([z2,y2,x2])
    cmap = px.colors.qualitative.Plotly
    cmap1 = [cmap[i%10] for i in idx1]
    cmap2 = [cmap[i%10] for i in idx2]
    labels1=['E'+str(i) for i in idx1]
    labels2=['E'+str(i) for i in idx2]
    fig = go.Figure(data=[go.Scatter3d(x=x1, y=y1, z=z1,
                                       mode='markers+text', 
                                       marker_color=cmap1),
                          go.Scatter3d(x=x_fine1,y=y_fine1,z=z_fine1, 
                                       mode='lines'),
                          go.Scatter3d(x=x2, y=y2, z=z2,
                                       mode='markers+text', 
                                       marker_color=cmap2),
                          go.Scatter3d(x=x_fine2,y=y_fine2,z=z_fine2, 
                                       mode='lines')])
    
    iplot(fig)
    
def plot_gpa_output(aligned_points, mean_points):
    
    scatters = []
    cmap = px.colors.qualitative.Plotly
    for point_id, point_set in enumerate(aligned_points):
        idx=np.arange(point_set.shape[0])
        qc_idx = point_set[:,3] != 0
        idx=idx[qc_idx]
        labels=['E'+str(i) for i in idx]
        
        cmap_points = [cmap[i%10] for i in idx]
        
        point_set_plot = point_set[qc_idx, 0:3]
        scatters.append(go.Scatter3d(x=point_set_plot[:,2], y=point_set_plot[:,1], z=point_set_plot[:,0], 
                                     mode='markers+lines', 
                                     marker_color=cmap_points,
                                     marker_size=5,
                                     opacity=0.3,
                                     name='Trace '+str(point_id),
                                     line=dict(color='#1f77b4',
                                               width=1)))
    
    mean_idx=np.arange(mean_points.shape[0])
    mean_qc = mean_points[:,3] != 0
    mean_idx=mean_idx[mean_qc]
    print(mean_idx)
    #mean_labels=['E'+str(i) for i in mean_idx]
    mean_cmap = [cmap[i%10] for i in mean_idx]
    mean_points_plot = mean_points[mean_idx, 0:3]

    mean_fig = scatters.append(go.Scatter3d(x=mean_points_plot[:,2], y=mean_points_plot[:,1], z=mean_points_plot[:,0], 
                                            mode='markers+lines', 
                                            marker_color=mean_cmap,
                                            name='Mean',
                                            line=dict(color='#ff7f0e', 
                                                      width=5)))
    
    fig = go.Figure(data=scatters)
    
    iplot(fig)
    
def plot_multi_points(list_of_points):
    scatters = []
    cmap = px.colors.qualitative.Light24
    for point_id, point_set in enumerate(list_of_points):
        idx=np.arange(point_set.shape[0])
        qc_idx = point_set[:,3] != 0
        idx=idx[qc_idx]
        labels=['E'+str(i) for i in idx]
        
        cmap_points = [cmap[i%24] for i in idx]
        
        point_set_plot = point_set[qc_idx, 0:3]
        scatters.append(go.Scatter3d(x=point_set_plot[:,2], y=point_set_plot[:,1], z=point_set_plot[:,0], 
                                     mode='markers+lines', 
                                     marker_color=cmap_points,
                                     marker_size=5,
                                     opacity=1,
                                     name='Trace '+str(point_id),
                                     line=dict(color=cmap[point_id],
                                               width=2)))
    fig = go.Figure(data=scatters)
    
    iplot(fig)