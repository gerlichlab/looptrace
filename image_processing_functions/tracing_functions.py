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
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import single, dendrogram
#from pycpd import RigidRegistration

def pwd_calc(traces):
    # Calculate pairwise distances for all traces.
    points = [points_from_df_nan(df)[0] for _, df in traces.groupby(level=0)]
    pwds = [cdist(p, p) for p in points]
    pwds = np.stack(pwds)
    return pwds

def tracing_length_qc(traces, pwds, min_length=0):
    grouped=traces.groupby(level=0)
    traces_long=grouped.filter(lambda x : x['QC'].sum()>=min_length)
    
    points = [points_from_df_nan(df)[0] for _, df in traces_long.groupby(level=0)]
    pwds = [cdist(p, p) for p in points]
    pwds_long = np.stack(pwds)
    
    return traces_long, pwds_long

def trace_analysis(traces, pwds):
    pairwise_trace_idx = list(itertools.combinations(traces.index.get_level_values(0).unique(),2))
    pairwise_pwd_idx = list(itertools.combinations(range(pwds.shape[0]),2))
    res = []
    for ((idx1, idx2), (idx_p1, idx_p2)) in zip(pairwise_trace_idx,pairwise_pwd_idx):
        A, A_idx, B, B_idx, B_aligned, B_aligned_idx, mse = rigid_transform_3D(traces.loc[idx1], traces.loc[idx2])
        pwd_mse = mat_corr_mse(pwds[idx_p1],pwds[idx_p2])
        pwd_pcc = mat_corr_pcc(pwds[idx_p1],pwds[idx_p2])        
        res.append([idx1, idx2, A, A_idx, B, B_idx, B_aligned, B_aligned_idx, mse, pwd_mse, pwd_pcc])
    columns=['idx1', 'idx2', 'A', 'A_idx', 'B', 'B_idx', 'B_aligned', 'B_aligned_idx', 'aligned_mse', 'pwd_mse', 'pwd_pcc']
    output=pd.DataFrame(res,columns=columns)
    return output

def trace_clustering(paired, metric):
    labels=np.unique(np.concatenate((paired['idx1'],paired['idx2'])))
    if metric == 'aligned_mse' or metric == 'pwd_mse':
        Z=single(paired[metric])
    elif metric == 'pwd_pcc':
        Z=single(1-paired[metric])
    else:
        raise ValueError('Inappropriate metric.')
    
    dendrogram(Z, labels=labels)
    return Z

def rigid_transform_3D(df_A, df_B):
    A_orig, A_orig_idx = points_from_df(df_A)
    B_orig, B_orig_idx = points_from_df(df_B)
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

    # dot is matrix multiplication for array
    H = np.matmul(Am, Bm.T)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T,U.T)

    # special reflection case 
    if np.linalg.det(R) < 0:
        #print("det(R) < R, reflection detected!, correcting for it ...\n");
        Vt[2,:] *= -1
        R = np.matmul(Vt.T,U.T)

    t = Bc - np.matmul(R,Ac)
    B_ = np.matmul(R.T,B-t)
    mse = ((A - B_)**2).mean()
    B_aligned = np.matmul(R.T,B_orig.T-t).T
    return A_orig, A_orig_idx, B_orig, B_orig_idx, B_aligned, B_orig_idx, mse

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
    _traces = trace_df[trace_df['QC']==1]
    points = np.array([_traces['x'].values,_traces['y'].values,_traces['z'].values]).T
    idx = _traces.index.get_level_values('frame')
    return points, idx

def points_from_df_nan(trace_df):
    traces=trace_df.copy()
    traces[traces['QC']==0] = np.nan
    points = np.array([traces['x'].values,traces['y'].values,traces['z'].values]).T
    idx = traces.index.get_level_values('frame')
    return points, idx

def matching_points_from_dfs(df_A,df_B):
    df_A_idx=df_A[df_A['QC']==1].index.get_level_values('frame')
    df_B_idx=df_B[df_B['QC']==1].index.get_level_values('frame')
    idx=df_A_idx.intersection(df_B_idx)
    i = pd.IndexSlice
    points_A, idx_A = points_from_df(df_A.loc[i[:, :, :, idx], i[:]])
    points_B, idx_B = points_from_df(df_B.loc[i[:, :, :, idx], i[:]])
    return points_A, idx_A, points_B, idx_B

def spline_interp(points):
    tck, u = interpolate.splprep(points, s=0)
    knots = interpolate.splev(tck[0], tck)
    u_fine = np.linspace(0,1,num=100)
    fine = interpolate.splev(u_fine, tck)
    return fine

def mat_corr_mse(mat1, mat2):
    mse = np.nanmean((mat1-mat2)**2)
    return mse

def mat_corr_pcc(mat1,mat2):
    mat1_bar=np.nanmean(mat1)
    mat2_bar=np.nanmean(mat2)
    
    pcc_num=np.nansum((mat1-mat1_bar)*(mat2-mat2_bar))
    pcc_denom=np.sqrt(np.nansum((mat1-mat1_bar)**2)*np.nansum((mat2-mat2_bar)**2))
    pcc=pcc_num/pcc_denom
    
    return pcc
    
def plot_trace(points, idx):
    points=points.T
    z,y,x=points
    z_fine,y_fine,x_fine=spline_interp(points)

    #fig1 = plt.figure(1)
    #ax3d = fig1.add_subplot(111, projection='3d')
    cmap = matplotlib.cm.get_cmap('tab10')
    #scatter = ax3d.scatter(points[2], points[1], points[0], '.', color=cmap(idx), alpha=1, lw=3)
    #scatter_proxy = matplotlib.lines.Line2D(points[0],points[1], linestyle="none", c=cmap(idx), marker = 'o')
    #ax3d.legend(scatter_proxy, loc="lower left", title="Seq")
    #ax3d.plot(x_knots, y_knots, z_knots, 'go')
    #ax3d.plot(x_fine, y_fine, z_fine, '-', lw=3)
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                   mode='markers', marker_color=cmap(idx)),
                          go.Scatter3d(x=x_fine,y=y_fine,z=z_fine, 
                                       mode='lines')])
    iplot(fig, auto_open=True)
    
def plot_traces(traces,idx):
    if type(idx) == int:
        idx=[idx]
    trace_df=traces[traces['QC']==1]
    trace_df_sel=pd.concat([trace_df.xs(i) for i in idx])
    trace_index=trace_df_sel.index.get_level_values('frame')
    
    labels=['E'+str(t) for t in trace_index]
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

'''
def plot_traces_sub(traces,idx=False):
    if idx==False:
        idx=traces.index.levels[0]
    
    fig = px.scatter_3d(traces, x='x', y='y', z='z',
                        facet_row='')
    plot(fig, auto_open=True)
'''
    
def plot_double_trace(points1, idx1, points2, idx2):
    points1=points1.T
    points2=points2.T
    z1,y1,x1=points1
    z2,y2,x2=points2
    z_fine1,y_fine1,x_fine1=spline_interp(points1)
    z_fine2,y_fine2,x_fine2=spline_interp(points2)
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
    
def plot_paired_traces(pair_df, idx):
    points1=pair_df['A'][idx]
    idx1=pair_df['A_idx'][idx]
    points2=pair_df['B_aligned'][idx]
    idx2=pair_df['B_aligned_idx'][idx]
    plot_double_trace(points1, idx1, points2, idx2)
    
