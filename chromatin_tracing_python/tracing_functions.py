# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 09:19:04 2020

@author: ellenberg
"""
import itertools
import pandas as pd
import numpy as np
import matplotlib
from plotly.offline import iplot
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import interpolate
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import napari
from joblib import Parallel, delayed
#from pycpd import RigidRegistration

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

    columns=['idx1', 'idx2', 'A', 'A_idx', 'B', 'B_idx', 'B_aligned', 'B_aligned_idx', 'aligned_mse', 'aligned_pcc', 'pwd_mse', 'pwd_pcc']
    output=pd.DataFrame(res,columns=columns)
    return output

def single_trace_analysis(traces, pwds, idx1, idx2, idx_p1, idx_p2):
    A, A_idx, B, B_idx, B_aligned, B_aligned_idx, aligned_mse, aligned_pcc = rigid_transform_3D(traces.loc[idx1], traces.loc[idx2])
    pwd_mse = mat_corr_mse(pwds[idx_p1],pwds[idx_p2])
    pwd_pcc = mat_corr_pcc(pwds[idx_p1],pwds[idx_p2])
    return idx1, idx2, A, A_idx, B, B_idx, B_aligned, B_aligned_idx, aligned_mse, aligned_pcc, pwd_mse, pwd_pcc
    
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
        Z=linkage(paired[metric], method=method)
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

def rigid_transform_3D(df_A, df_B):
    '''
    Calculates rigid transformation of two 3d points sets based on:
    Least-squares fitting of two 3-D point sets. IEEE T Pattern Anal 1987
    DOI: 10.1109/TPAMI.1987.4767965
    Only uses points present in both traces for alignment.

    Parameters
    ----------
    df_A, df_B : pd DataFrames with single trace data.

    Returns
    -------
    Point coordinates and indexes of original and aligned traces,
    and MSE of alignment.
    '''
    
    # Get points from input dataframe.
    A_orig, A_orig_idx = points_from_df(df_A)
    B_orig, B_orig_idx = points_from_df(df_B)

    # Ensure only matching points are used.
    # TODO: This is quite slow, bulk of this whole function. Implementing something faster should be easy.
    A, idx_A, B, idx_B = matching_points_from_dfs(df_A,df_B)
    
    #Need column vectors for calculation
    A=A.T
    B=B.T
    num_rows, num_cols = A.shape;

    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

    [num_rows, num_cols] = B.shape;
    if num_rows != 3:
        raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

    # find mean column wise and reshape to column vector
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
    
    # Transform the matched B matrix to A to calculate aligmen parameters.
    B_ = np.matmul(R.T,B-t)
    mse = mat_corr_mse(A, B_)
    pcc = mat_corr_pcc(A, B_)
    
    # Transform the originial, unmatched B matrix. 
    B_aligned = np.matmul(R.T,B_orig.T-t).T

    return A_orig, A_orig_idx, B_orig, B_orig_idx, B_aligned, B_orig_idx, mse, pcc

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

def points_from_df(trace_df):
    '''
    Helper function to extract point coordinates from trace dataframe.
    Only points passing QC during tracing are returned.

    Parameters
    ----------
    trace_df : pd DataFrame with trace data.

    Returns
    -------
    points : Nx3 np array with trace coordinates.
    idx : List, Original index of trace coordinates.
    '''
    
    _traces = trace_df[trace_df['QC']==1]
    points = np.array([_traces['x'].values,_traces['y'].values,_traces['z'].values]).T
    idx = _traces['frame']
    return points, idx

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

def mat_corr_mse(mat1, mat2):
    '''
    Calculate mean squared error of two matrices, ignoring nans.
    '''
    
    mse = np.nanmean((mat1-mat2)**2)
    return mse

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
        
def plot_traces(traces,idx):
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
    
def plot_paired_traces(pair_df, idx):
    '''
    Helper function for plotting two aligned traces in one figure.
    Also plots spline interpolation between points for visualization.
    
    Parameters
    ----------
    pair_df : pd DataFrame with paired trace analysis, output of trace_analysis.
    idx : Int, index of pair from paired dataframe. 
    '''
    
    points1=pair_df['A'][idx]
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
    
