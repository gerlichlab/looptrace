# -*- coding: utf-8 -*-
"""
Created by:

Kai Sandvold Beckwith
Ellenberg group
EMBL Heidelberg

Colors scheme for convenience:
tab:blue : #1f77b4
tab:orange : #ff7f0e
tab:green : #2ca02c
tab:red : #d62728
tab:purple : #9467bd
tab:brown : #8c564b
tab:pink : #e377c2
tab:gray : #7f7f7f
tab:olive : #bcbd22
tab:cyan : #17becf
"""
import os
import itertools
<<<<<<< Updated upstream
import pandas as pd
import numpy as np
import plotly.colors
=======
#from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
import tqdm
>>>>>>> Stashed changes
import re
from plotly.offline import iplot
import plotly.graph_objs as go
import plotly.express as px
from scipy import interpolate
<<<<<<< Updated upstream
from scipy.spatial.distance import cdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import napari
=======
from scipy.spatial.distance import cdist, squareform, pdist
from scipy.spatial import ConvexHull
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt


>>>>>>> Stashed changes
from numba import jit, njit

from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

import seaborn as sns

<<<<<<< Updated upstream
=======
def gen_random_coil(g_dist, s_dist = 24.7, std_scaling = 0.5, deg=360, n_traces = 1000, sigma_noise = 2):
    traces = []
    for j in range(n_traces):
        L = [np.random.normal(np.sqrt(d)*s_dist, np.sqrt(d)*s_dist*std_scaling, 1) for d in g_dist]  # Calculate the step lengths, drawn from normal distribution according to paramters set above.
        N = len(L)+1
        #N = 1135//step_denom #Number of steps in each random walk
        #N = np.sum(g_dist).astype(int)//step_denom + 1
        R = deg*(np.random.rand(N)) #Get the polar angles between. Picks random angle between 0-deg 
        A = deg*(np.random.rand(N)) #Get the azimethal angels. Picks random angle between 0-deg

        #We need to keep track of the change in angle over subsequent positions so the origin isn't reset at every point.
        R = np.cumsum(R)
        A = np.cumsum(A) 
        
        #Initialize trace arrays
        trace_id = np.ones(N) * j
        frame = np.array(range(N))+1
        x = np.zeros(N)
        y = np.zeros(N)
        z = np.zeros(N)
        qc = np.ones(N)
        
        for i in range(1,N): #converting spherical coordinates to cartesian.
            x[i] = x[i-1] + L[i-1]*sin(radians(R[i]))*cos(radians(A[i]))
            y[i] = y[i-1] + L[i-1]*sin(radians(R[i]))*sin(radians(A[i]))
            z[i] = z[i-1] + L[i-1]*cos(radians(R[i]))
        x = x + np.random.normal(0, sigma_noise, size=x.shape)
        y = y + np.random.normal(0, sigma_noise, size=y.shape)
        z = z + np.random.normal(0, sigma_noise, size=z.shape)

        # Create 6xN dataframe in the format (trace_id, frame, x, y, z, QC)
        traces.append(np.column_stack((trace_id, frame, x, y, z, qc)))
        #sampling = np.cumsum(np.round([0]+g_dist)).astype(int)//step_denom
        #xyz_array = xyz_array[sampling,:]
#
    traces = np.concatenate(traces)
    traces = pd.DataFrame(traces)#.reset_index(drop=True)
    traces.columns = ['trace_id','frame','x','y','z','QC']
    traces = traces.astype({'trace_id': int, 'frame': int, 'QC': int})
    traces['frame_name']='H'+traces['frame'].astype(str).str.zfill(2)

    return traces

def loop_freq_vs_random(traces,traces_rw):
    
    
    pwds = pwd_calc(traces)
    pwds_rw = pwd_calc(traces_rw)
    idx = np.random.choice(np.arange(pwds_rw.shape[0]), pwds.shape[0], replace=False)
    pwds_rw=pwds_rw[idx, :, :]
    print(pwds.shape, pwds_rw.shape)
    freq_exp = np.nansum(pwds<150, axis=0)/np.nansum(pwds>0, axis=0)
    freq_rw = np.nansum(pwds_rw<150, axis=0)/np.nansum(pwds_rw>0, axis=0)
    freq = np.isfinite(freq_exp/freq_rw) & (freq_exp/freq_rw > 1)
    #print(freq)
    freq_ss = np.nansum(np.triu((pwds<150) * freq, 1), axis=(1,2))
    freq_base = np.nansum(np.triu((pwds<150), 1), axis=(1,2))
    freq_rw = np.nansum(np.triu((pwds_rw<150) & np.isfinite(pwds), 1), axis=(1,2))
    return freq_ss, freq_base, freq_rw

def pylochrom_coords_to_traces(coords):
    N_traces, N_steps, _ = coords.shape
    traces = []
    for i in range(N_traces):
        trace = {}
        trace['z'] = coords[i,:,0]
        trace['y'] = coords[i,:,1]
        trace['x'] = coords[i,:,2]
        trace['frame'] = list(range(N_steps))
        trace['trace_id'] = [i]*N_steps 
        trace['QC'] = [1]*N_steps
        #print(pd.DataFrame(trace))
        traces.append(pd.DataFrame(trace))
    return pd.concat(traces)

>>>>>>> Stashed changes
def tracing_qc(traces, qc_config):
    df = traces.copy()
    A_to_BG = qc_config['A_to_BG']
    sigma_xy_max = qc_config['sigma_xy_max']
    sigma_z_max = qc_config['sigma_z_max']
    max_dist = qc_config['max_dist']

    qc = np.ones((len(traces)), dtype=bool)

    if max_dist:
        refs = df[df['frame'] == df['ref_frame']]
        for dim in ['z','y','x']:
            refs_map = dict(zip(refs['trace_id'], refs[dim]))
            df[dim+'_ref'] = df['trace_id'].map(refs_map)
        ref_dist = np.sqrt((df['z_ref']-df['z'])**2 + (df['y_ref']-df['y'])**2 + (df['x_ref']-df['x'])**2)
        qc = qc & (ref_dist < max_dist)
    
    qc = qc & (df['A'] > (A_to_BG*df['BG']))
    qc = qc & (df['sigma_xy'] < sigma_xy_max)
    qc = qc & (df['sigma_z'] < sigma_z_max)
    qc = qc & df['z_px'].between(0,100)
    qc = qc & df['y_px'].between(0,100)
    qc = qc & df['x_px'].between(0,100)

    return qc
'''
def tracing_qc(row, qc_dict, traces_df=None):

    
    A_to_BG = qc_dict['A_to_BG']
    sigma_xy_max = qc_dict['sigma_xy_max']
    sigma_z_max = qc_dict['sigma_z_max']
    max_dist = qc_dict['max_dist']
    

    if max_dist:
        ref_frame = row['ref_frame'] #
        #trace_id = row['trace_id']
        #ref_frame = traces_df.query('trace_id == @trace_id').iloc[ref]
        z_c = ref_frame['z']
        y_c = ref_frame['y']
        x_c = ref_frame['x']
        z = row['z']
        y = row['y']
        x = row['x']

        dist = ((z-z_c)**2 + (y-y_c)**2 + (x-x_c)**2)**0.5

        if dist > max_dist:
            return 0

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
'''

def view_context(all_images,
                 contrast= ((0,5000),(0,2000),(0,5000)),
                 trace_id = None, 
                 ref_slice = None,
                 rois = None):
    
    '''
    Convenvience function to view a given ROI in context in napari.
    If not trace_id or ROIs given the whole image stack divided by channel is shown.
    '''
    import napari
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
                                            size=10,
                                            edge_width=3,
                                            edge_color='red',
                                            face_color='transparent',
                                            n_dimensional=True)
            sel_dim = np.concatenate([[int(ref_slice)], point.astype(int)])

            for dim in range(len(sel_dim)):
                viewer.dims.set_current_step(dim, sel_dim[dim])
    return viewer

def view_fits(traces, imgs, mode='2D', contrast=(100,10000), axis=2):
    '''
    Convenience function to view 3d guassian fits on top of 2D (max z-projection) or 3D spot data.
    '''
    import napari
    points = traces[['trace_id', 'frame', 'z_px', 'y_px', 'x_px', 'QC']].to_numpy()
    points = points[points[:,5].astype(bool),0:5]
    print
    if mode == '2D':
        imgs=np.max(imgs, axis=axis)
        with napari.gui_qt():
            viewer = napari.view_image(imgs, contrast_limits=contrast)
            viewer.add_points(points[:,(0,1,3,4)], size=[0,0,1,1], face_color='blue', symbol='cross', n_dimensional=False)
    elif mode == '3D':
        with napari.gui_qt():
            viewer = napari.view_image(imgs, contrast_limits=contrast)
            viewer.add_points(points, size=[0,0,3,1,1], face_color='blue', symbol='cross', n_dimensional=True)

def points_for_overlay(traces, rois, config):
    '''
    Generate the fit coordinates in a format convenient to display as a marker in napari.

    roi_image_size = config['roi_image_size']
    points_df = traces.copy()
    for i, roi in rois.iterrows():
        #transp_z=(roi_image_size[0]-(roi['z_max']-roi['z_min']))//2
        #transp_y=(roi_image_size[1]-(roi['y_max']-roi['y_min']))//2
        #transp_x=(roi_image_size[2]-(roi['x_max']-roi['x_min']))//2
        idx = traces['trace_id'] == roi.name
        points_df[idx] = points_df[idx].assign(z_px = traces[idx]['z_px'],
                                               y_px = traces[idx]['y_px'],
                                               x_px = traces[idx]['x_px'])

    #points_df[['y_px','x_px']]=points_df[['y_px','x_px']].clip(lower=0, upper=64)
    #points_df[['z_px']]=points_df[['z_px']].clip(lower=0, upper=16)
    points=points_df[['trace_id', 'frame', 'z_px', 'y_px', 'x_px']].to_numpy()
    '''
    
    return points

def euclidean_dist(traces, frame_names, column = 'frame_name'):
    '''Calculate the eucledian distances between the positions indicated in all traces.

    Args:
        traces (DataFrame): Trace data
        frame_names (list): List of names the two positions to compare.

    Returns:
        df_sel (DataFrame): Eucledian distances per trace of the two positions
    '''

<<<<<<< Updated upstream
    df_sel = traces[traces['frame_name'].isin(frame_names)]
    df_qc = df_sel.groupby(['trace_id'])[['QC']].sum()
=======
    df_sel = traces[traces[column].isin(frame_names)]
    trace_ids = df_sel['trace_id'].unique()
    qc = df_sel.groupby(['trace_id'])[['QC']].sum()['QC'].values
>>>>>>> Stashed changes
    df_sel = df_sel.groupby(['trace_id'])[['z','y','x']].diff().dropna()
    df_sel['euclidean'] = ((df_sel['z'])**2 + df_sel['y']**2 + df_sel['x']**2)**0.5
    df_sel['euclidean_res'] = ((0.5*df_sel['z'])**2 + df_sel['y']**2 + df_sel['x']**2)**0.5
    df_sel.index = df_qc.index
    df_sel['QC'] = df_qc['QC']
    df_sel = df_sel[df_qc['QC'] == 2]
    df_sel['id'] = str(frame_names)
    df_sel=df_sel.reset_index()
    return df_sel

def eucledian_dist_all(traces):
    '''Calculate eucledian distances for all pairwise combinations of positions

    Args:
        traces (DataFrame): Trace data

    Returns:
        df (DataFrame): DataFrame with all the pairwise eucledian distances.
    '''
    frame_names = traces['frame_name'].unique()
    combos = itertools.combinations(frame_names, 2)
    df = []
    for c in combos:
        df.append(euclidean_dist(traces, c))
    df = pd.concat(df)
    return df

def genomic_distance_map(genomic_positions):
    '''
    Generate mapping of position combination names to genomic distance
    from dict of the format {'frame_name':genomic_position}.
    Used for calculating genomic vs spatial distances.
    '''

    combos = itertools.combinations(genomic_positions.keys(), 2)
    g_dists = {}
    for c in combos:
        dist = genomic_positions[c[1]]-genomic_positions[c[0]]
        g_dists[str(c)] = dist
    return g_dists

def pwd_calc(traces):
    '''
    Parameters
    ----------
    traces : pd DataFrame with trace data.

    Returns
    -------
    pwds : Pair-wise distance matrixes for traces as an 3D numpy array.
    '''
    
    points = points_from_traces_nan(traces, trace_ids = -1)
    pwds = [cdist(p, p) for p in points]
    pwds = np.stack(pwds)
    return pwds

def tracing_length_qc(traces, min_length=0):
    '''
    Parameters
    ----------
    traces : pd DataFrame with trace data.
    min_length : Int, minimum length of trace to pass QC

    Returns
    -------
    traces : pd DataFrame with shorter traces removed.
    pwds : 3-dim np array of pwds that passed length QC.

    '''
    grouped=traces.groupby('trace_id')
    traces_long=grouped.filter(lambda x : x['QC'].sum()>=min_length)
    return traces_long

def trace_analysis(traces, pwds):
    '''
    Calculates pairwise trace similarity based on :
        - MSE of points after rigid alignment
        - PCC of points after rigid alignment
        - MSE of pairwise distance matrices
        - PCC of pwds.
        
    Parameters
    ----------
    traces : pd DataFrame with trace data.
    pwds : 3D np array from pwd_calc.

    Returns
    -------
    output : pd DataFrame with pairwise similarity results, including indexes of traces.
    '''

    points = np.stack(points_from_traces(traces))
    trace_idx = traces.trace_id.unique()
    res = trace_analysis_loop(points, pwds, trace_idx)
    #pairwise_trace_idx = list(itertools.combinations(traces['trace_id'].unique(),2))
    #pairwise_pwd_idx = list(itertools.combinations(range(pwds.shape[0]),2))
    #res = Parallel(n_jobs=-2)(delayed(single_trace_analysis)
    #                          (traces, pwds, idx1, idx2, idx_p1, idx_p2) for 
    #                          ((idx1, idx2), (idx_p1, idx_p2)) in 
    #                          zip(pairwise_trace_idx,pairwise_pwd_idx))

    columns=['idx1', 'idx2', 'aligned_mse', 'aligned_pcc', 'pwd_mse', 'pwd_pcc']
    output=pd.DataFrame(res,columns=columns)
    output[['idx1', 'idx2']] = output[['idx1', 'idx2']].astype(int)
    return output

@njit
def trace_analysis_loop(points, pwds, trace_idx):
    res = []
    idx = list(range(len(points)))
    for i in idx[:-1]:
        a = points[i]
        d_1 = pwds[i]
        for j in idx[i+1:]:
            b = points[j]
            d_2 = pwds[j]
            out = list(single_trace_analysis(a,b,d_1,d_2))
            res.append([trace_idx[i], trace_idx[j]] + out)
    return res

def compare_trace_analysis(traces1, traces2, pwds1, pwds2):
    points1 = points_from_traces(traces1)
    points2 = points_from_traces(traces2)
    trace_idx1 = list(traces1.trace_id.unique())
    trace_idx2 = list(traces2.trace_id.unique())
    idx1 = list(range(len(points1)))
    idx2 = list(range(len(points2)))
    res = []

    for i in idx1:
        a = points1[i]
        d_1 = pwds1[i]
        for j in idx2:
            b = points2[j]
            d_2 = pwds2[j]
            out = list(single_trace_analysis(a,b,d_1,d_2))
            res.append([trace_idx1[i], trace_idx2[j]] + out)    

    columns=['idx1', 'idx2', 'aligned_mse', 'aligned_pcc', 'pwd_mse', 'pwd_pcc']
    output=pd.DataFrame(res,columns=columns)
    return output

@njit
def single_trace_analysis(a,b,d_1,d_2):
    '''
    Perform pairwise analysis of two single traces according to MSE and PCC metrics
    of their aligned points and their distance matrices.
    
    Parameters
    ----------
    traces : pd DataFrame with trace data.
    pwds : 3D np array from pwd_calc.
    idx1: int, trace_id of first trace
    idx2: int, trace_id of second trace
    idx_p1: int, index of first trace in pwd matrix
    idx_p2: int, index of second trace in pwd matrix

    Returns
    -------
    output : The input trace_ids, and the similiarity metrics of the registered traces.
    '''
    #Get points by their trace indices.
    a, b = match_two_pointsets(a, b)
    
    #Center the pointsand rescale to avoid issues of large numbers for PCC calculation.
    a = a-numba_mean_axis0(a)
    b = b-numba_mean_axis0(b)

    #Align the point sets
    b_reg = rigid_transform_3D(b, a, prematch=True)

    #Calculate distances between the point sets
    aligned_mse = euclidean(a, b_reg)
    aligned_pcc = 1-mat_corr_pcc(a, b_reg)
    
    #Calculate distances between the point distance matrices
    pwd_mse = euclidean(d_1,d_2)
    #rescale data to avoid issues of large numbers in PCC calculation
    pwd_pcc = 1-mat_corr_pcc(d_1/1000,d_2/1000)

    return aligned_mse, aligned_pcc, pwd_mse, pwd_pcc

def trace_clustering(pairs, metric='pwd_pcc', dendro_method='single', color_threshold=None):
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
    cluster_df : Dataframe with results of
                 hierarchial clustering (see scipy.cluster.hierarchy.single docs)
    '''
    from scipy.cluster.hierarchy import set_link_color_palette
    cmap = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']#px.colors.qualitative.Plotly
    #cmap=cmap[5:]
    set_link_color_palette(cmap)
    labels=list(np.unique(np.concatenate((pairs['idx1'],pairs['idx2']))))

    Z=linkage(pairs[metric], method=dendro_method)

    fig1 = plt.figure(figsize=(20,20))
    dendro=dendrogram(Z, labels=labels, color_threshold=color_threshold, leaf_font_size=9)
    if color_threshold is None:
        color_threshold = 0.7*max(Z[:,2])
    clusters=fcluster(Z, color_threshold, criterion='distance')
    cluster_df=pd.DataFrame([labels, clusters]).T 
    cluster_df.columns=['trace_id', 'dendro']
    return cluster_df

def further_trace_clustering(pairs, cluster_df, metric, n_clusters=5):
    '''Perform a variety of sklearn clustering metrics on the data.

    Args:
        pairs (DataFrame): Distance matrix of trace pairs
        cluster_df (DataFram): Existing clustering dataframe form dendrogram clustering
        metric (str): Which metric to use as distance, typically aligned_pcc
        n_clusters (int, optional): Number of predefined clusters. Defaults to 5.

    Returns:
        cluster_df (DataFrame): Updated clustering dataframe with additional clustering of all traces
        pos (ndarray): Positions of traces in 2D coordinates generated by MDS. 
    '''
    from sklearn.manifold import MDS
    from sklearn import cluster

    distances = squareform(pairs[metric])
    embedding = MDS(n_components=2, dissimilarity='precomputed')
    pos = embedding.fit_transform(distances)

    affinity_propagation = cluster.AffinityPropagation(damping=0.9, preference=-400)
    aff = affinity_propagation.fit_predict(distances)
    cluster_df['affinity'] = aff
    spectral = cluster.SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity="precomputed")
    spec = spectral.fit_predict(1-distances)
    cluster_df['spectral'] = spec
    kmeans = cluster.KMeans(n_clusters=n_clusters)
    km = kmeans.fit_predict(distances)
    cluster_df['kmeans'] = km
    ward_model = cluster.AgglomerativeClustering(n_clusters=n_clusters)
    ward = ward_model.fit_predict(distances)
    cluster_df['ward'] = ward

    return cluster_df, pos

def cluster_similarity(traces, cluster_df, method='cluster', metric='aligned_pcc'):
    '''Function to compare similarity of traces within and between clusters.
    #TODO: Hardcoded to aligned_pcc metric, fix this    

    Args:
        traces (DataFrame): Trace data
        cluster_df (DataFrame): Cluster assignment of traces
        metric (str, optional): Which cluster method from the clustering assigment to use. Defaults to 'cluster'.

    Returns:
        (list): List with the mean values of the between and within cluster similarities.
    '''


    clust_ids = sorted(cluster_df[method].unique())
    clust_pairs = list(itertools.combinations(clust_ids,2))
    res_combo = []
    for i, j in clust_pairs:
        clust1 = sorted(cluster_df.query('{0} == {1}'.format(method, i)).trace_id.unique())
        clust2 = sorted(cluster_df.query('{0} == {1}'.format(method, j)).trace_id.unique())
        traces1 = traces[traces['trace_id'].isin(clust1)]
        traces2 = traces[traces['trace_id'].isin(clust2)]
        pwds1 = pwd_calc(traces1)
        pwds2 = pwd_calc(traces2)

        pairs_1_2 = compare_trace_analysis(traces1, traces2, pwds1, pwds2)
        res_combo.append(pairs_1_2[metric].mean())

    res_single = []
    for i in clust_ids:
        clust_i = sorted(cluster_df.query('{0} == {1}'.format(method, i)).trace_id.unique())
        traces_i = traces[traces['trace_id'].isin(clust_i)]
        pwds_i = pwd_calc(traces_i)
        pairs_i = trace_analysis(traces_i, pwds_i)
        res_single.append(pairs_i[metric].mean())
    
    return [np.mean(res_combo), np.mean(res_single)]


def run_gpa_all_clusters(traces, cluster_df, metric='dendro', min_cluster = 1):
    '''
    Running function to perform GPA analysis on all clusters identified in trace_clustering()
    with number of members above min_cluster.
    '''

    #Find unique cluster IDs from clustering table.
    cluster_ids=set(cluster_df[metric])
    #Generate list of lists of all cluster members over min_cluster length.
    all_cluster_members = []
    for cluster_id in cluster_ids:
        cluster_members = cluster_df[cluster_df[metric]==cluster_id]['trace_id'].values
        if len(cluster_members)>=min_cluster:
            all_cluster_members.append(cluster_members)
            print(f'Cluster ID {cluster_id}, members: {cluster_members}.')

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

def general_procrustes_analysis(traces, trace_ids='all', crit=0.01):
    '''
    General procrustes analysis is performed as described in e.g.
    https://en.wikipedia.org/wiki/Generalized_Procrustes_analysis
    Runs until change in procrustes distance is less than crit.

    Returns all the aligned traces, the mean trace and the std of all the aligned traces.
    '''
    if isinstance(trace_ids, str):
        trace_ids = list(traces['trace_id'].unique())
    elif isinstance(trace_ids, list):
        pass
    else:
        trace_ids=list(trace_ids.astype(int))
    # Make list of all points of selected traces
    all_points = points_from_traces(traces, trace_ids)
    # Select a random template for initial loop
    #np.random.seed(1)
    t_idx = np.random.randint(0,len(all_points))
    template = all_points[t_idx]
    template = center_points_qc(template)
    #The initial distance before alignment.
    prev_dist = np.sum([procrustes_dist(template, points) for 
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
        
    print(f'GPA converged after {n_cycles} cycles with distance {dist}.')
    
    #Calculate standard deviation of all points:
    points_std = np.nanstd(np.stack(all_points), axis = 0)

    return all_points, points_mean, points_std

def piecewise_gpa(traces, trace_ids='all', crit=0.01, segment_length = 5, overlap = 3):
    '''
    General procrustes analysis is performed as described in e.g.
    https://en.wikipedia.org/wiki/Generalized_Procrustes_analysis
    Runs until change in procrustes distance is less than crit.

    Returns all the aligned traces, the mean trace and the std of all the aligned traces.
    '''
    if trace_ids == 'all':
        trace_ids = list(traces['trace_id'].unique())
    else:
        trace_ids=list(trace_ids.astype(int))
    # Make list of all points of selected traces
    hybs = np.sort(traces.query('QC == 1').hyb.unique())
    segments = [hybs[i*(segment_length-overlap):i*(segment_length-overlap)+segment_length] for i in range(0,len(hybs)//(segment_length-overlap))]
    segments = [s for s in segments if len(s) == segment_length]
    print(segments)
    aligned_segments = []
    for seg in tqdm.tqdm(segments):
        a = seg[0]
        b = seg[-1]
        traces_seg = traces.query('hyb >= @a & hyb <= @b')
        aligned_segments.append(general_procrustes_analysis(traces_seg, trace_ids='all', crit=0.01, template_points = segment_length-1)[1])
    
    full_trace = np.zeros((len(hybs), 4))
    full_trace[0:segment_length, :] = aligned_segments[0]
    for i in np.arange(0,len(aligned_segments)-1):
        print(i)
        prev_seg = full_trace[i*(segment_length-overlap):i*(segment_length-overlap) + segment_length].copy()
        prev_seg[0:overlap-1,3] = 0
        next_seg = aligned_segments[i+1].copy()
        next_seg[-overlap+1:,3] = 0
        reg_segment = rigid_transform_3D(next_seg,prev_seg,prematch=False)
        new_avg = np.mean([prev_seg[-overlap:,:],reg_segment[:overlap,:]], axis=0)
        next_seg[:overlap,:] = new_avg
        next_seg[:,3] = 1
        full_trace[(i+1)*(segment_length-overlap):(i+1)*(segment_length-overlap)+segment_length, :] = next_seg
        
    return full_trace
        


def general_procrustes_loop(all_points, template):
    '''
    A single cycle in the general procrustes analysis.
    Returns the points in all_points aligned to template,
    the mean points and the procrustes distance to the mean.
    '''
    # Align all point sets to mean template
    all_points_aligned = [rigid_transform_3D(offset, template) for 
                          offset in all_points]
    all_points_aligned = [points for points in all_points_aligned if points.shape[0] >3]#Ensure at least 3 points in traces.
    #Set values that do not pass QC to nan in new list.
    all_points_aligned_qc=[]
    for points in all_points_aligned:
        points[points[:,3] == 0, 0:3]=np.nan
        all_points_aligned_qc.append(points)
    
    #Calculate mean points from QC list ignoring nans.:
    #points_mean = numba_trimmean_axis0(np.stack(all_points_aligned_qc), proportiontocut=0.1)
    points_mean = np.nanmean(np.stack(all_points_aligned_qc), axis=0)
    #points_mean = np.nanmedian(np.stack(all_points_aligned_qc), axis=0)
    #The "QC" for the mean is 1 if at least one element has a QC=1
    points_mean[:,3]=1#np.ceil(points_mean[:,3])
    # Calculate distance to mean:
    dist = np.sum([procrustes_dist(points_mean, points) for 
                   points in all_points_aligned])
    return all_points_aligned, points_mean, dist

@njit
def procrustes_dist(a, b):
    '''
    Procrustes distance (identical to RMSD) between two point sets.
    Matches them before calculation.
    '''
<<<<<<< Updated upstream
    a, b = match_two_pointsets(a, b)    
    dist = np.sqrt(np.mean((a-b)**2))
    return dist
=======
    #print(a,b)
    a, b = match_two_pointsets(a, b)
    #print(a,b)
    if a.shape[0] == 0:
        return 1e6    
    else:
        dist = np.sqrt(np.mean((a-b)**2))
        return dist
>>>>>>> Stashed changes

@njit
def procrustes_dist_corr(a, b):
    '''
    Correlation (PCC) between two point sets.
    Matches them before calculation.
    '''
    a, b = match_two_pointsets(a, b)    
    dist = 1-mat_corr_pcc(a,b)
    return dist

@njit
def numba_mean_axis0(arr):
    '''Helper function due to lack of numba support for axis arguments.
    '''

    return np.array([np.mean(arr[:,i]) for i in range(arr.shape[1])])

@njit
def numba_trimmean_axis0(arr, proportiontocut=0.1):
    '''Helper function for a trimmed mean due to lack of numba support for a trimmed mean function.

    Args:
        arr (ndarray): Array to calculate
        proportiontocut (float, optional): [description]. Defaults to 0.1.

    Returns:
        [type]: [description]
    '''

    N = arr.shape[1]
    D = arr.shape[2]
    res = []

    for i in range(N):
        for j in range(D):
            a = arr[:,i,j]
            a = np.sort(a[~np.isnan(a)])
            low = np.round(a.size*proportiontocut)
            high = a.size - low
            res.append(np.mean(a[low:high]))
    res = np.array(res).reshape(N,D)
    return res

@njit
def rigid_transform_3D(A_orig, B_orig, prematch = False):
    '''
    Calculates rigid transformation of two 3d points sets based on:
    Least-squares fitting of two 3-D point sets. IEEE T Pattern Anal 1987
    DOI: 10.1109/TPAMI.1987.4767965
    
    Finds the optimal (lest squares) of B = RA + t, so mapping of A onto B.
    
    Modified from http://nghiaho.com/?page_id=671
    
    Only uses points present in both traces for alignment.

    Parameters
    ----------
    A, B : Nx4 (ZYX + QC) numpy ndarrays.

    Returns
    -------
    Coordinates of registered and transformed A
    '''
    
    #Ensure matching points
    if not prematch:
        A, B = match_two_pointsets(A_orig, B_orig)
    else:
        A, B = A_orig, B_orig
    if A.shape[0] == 0: #No matching points.
        return A
    # Subtract mean
    # Workaround for lack of "axis" argument support in numba:
    Ac = numba_mean_axis0(A)
    Bc = numba_mean_axis0(B)
    #Ac = np.mean(A, axis=0)
    #Bc = np.mean(B, axis=0)
    Am = A - Ac
    Bm = B - Ac

    # Calculate covariance matrix
    H = Am.T @ Bm

    # Find optimal rotation by SVD of the covariance matrix.
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T 

    # Handle case if the rotation matrix is reflected.
    if np.linalg.det(R) < 0:
        #print("det(R) < R, reflection detected!, correcting for it ...\n");
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    # calculate translation.
    t = Bc - R @ Ac

    #Transform the original vector with QC values
    A_reg = np.copy(A_orig)
    A_reg[:,:3] = (R @ A_orig[:,:3].T).T+t
    
    return A_reg

@jit
def match_two_pointsets(A, B):
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

    match_idx = A[:,3] * B[:,3] != 0
    A_match = A[match_idx,0:3]
    B_match = B[match_idx,0:3]
    return A_match, B_match

def points_from_traces(traces, trace_ids=-1):
    '''
    Helper function to extract point coordinates from trace dataframe.

    Parameters
    ----------
    traces : pd DataFrame with trace data.
    trace_ids: single or multiple trace_ids to extract

    Returns
    -------
    points_qc : list of  Nx4 np array with trace coordinates and QC value.
    '''
    
    arr = traces[['trace_id', 'z', 'y', 'x', 'QC']].to_numpy()
    
    
    if trace_ids == -1:
        trace_ids, idx = np.unique(arr[:,0], return_index=True)
        return np.split(arr, idx[1:], axis=0)

    elif not isinstance(trace_ids, (list, tuple)):
        trace_ids = [trace_ids]

    return [arr[arr[:,0] == i][:,1:] for i in trace_ids]

def points_from_traces_qc_filt(traces, trace_ids=-1):
    '''
    Helper function to extract point coordinates from trace dataframe.
    All points are returned, but points not passing QC during tracing 
    are not returned.   

    Parameters
    ----------
    trace_df : pd DataFrame with trace data.

    Returns
    -------
    points : Nx3 np array with trace coordinates.
    '''


    arr = traces[['trace_id', 'z', 'y', 'x', 'QC']].to_numpy()
    qc_idx = arr[:,4] == 1
    arr = arr[qc_idx,0:4]

    if trace_ids == -1:
        trace_ids, idx = np.unique(arr[:,0], return_index=True)
        return np.split(arr, idx[1:], axis=0)

    else:
        if not isinstance(trace_ids, (list, tuple)):
            trace_ids = [trace_ids]
        return [arr[arr[:,0] == id][:,1:4] for id in trace_ids]

def points_from_traces_nan(traces, trace_ids=-1):
    '''
    Helper function to extract point coordinates from trace dataframe.
    All points are returned, but points not passing QC during tracing 
    are returned as NaN.   

    Parameters
    ----------
    trace_df : pd DataFrame with trace data.

    Returns
    -------
    points : Nx3 np array with trace coordinates, NaN row returned if point did not pass QC.
    '''

    arr = traces[['trace_id', 'z', 'y', 'x', 'QC']].to_numpy()
    qc_idx = arr[:,4] == 1
    arr[~qc_idx,1:4] = np.nan

    if trace_ids == -1:
        trace_ids, idx = np.unique(arr[:,0], return_index=True)
        return np.split(arr[:,1:4], idx[1:], axis=0)

    elif not isinstance(trace_ids, (list, tuple)):
        trace_ids = [trace_ids]

    return [arr[arr[:,0] == id][:,1:4] for id in trace_ids]



def center_points_qc(a):
    qc = a[:,3] # QC of points
    ac = a[:,:3]-np.mean(a[qc>0,:3], axis=0) #Subtract mean of QC==1 points
    ac = ac + np.abs(np.min(ac[qc>0,:3], axis=0)) #Shift so all values positive
    qc = qc[:,np.newaxis] #Reshape QC to reappend
    return np.append(ac,qc,axis=1)


def spline_interp(points, n_points=100):
    '''
    Performs cubic B-spline interpolation on point coordinates.

    Parameters
    ----------
    points : n_points X ndim list of point coordinates.

    Returns
    -------
    fine : 100 X ndim nd array of interpolated points.

    '''
    
    tck, u = interpolate.splprep(points, s=0, k=2)
    #knots = interpolate.splev(tck[0], tck)
    u_fine = np.linspace(0,1,num=n_points)
    fine = interpolate.splev(u_fine, tck)
    return fine

@jit(nopython=True)
def euclidean(a, b):
    return np.sqrt(np.nansum((a-b)**2))

@jit(nopython=True)
def mat_corr_rmse(a, b):
    '''
    Calculate mean squared error of two matrices, ignoring nans.
    '''
    rmse = np.sqrt(np.nanmean((a-b)**2))
    return rmse

@jit(nopython=True)
def mat_corr_pcc(a,b):
    '''
    Calculate pearson's corr coef of two matrices, ignoring nans.
    '''
    a_m=np.nanmean(a)
    b_m=np.nanmean(b)
    
    pcc_num=np.nansum((a-a_m)*(b-b_m))
    pcc_denom=np.sqrt(np.nansum((a-a_m)**2))*np.sqrt(np.nansum((b-b_m)**2))
    pcc=np.divide(pcc_num,pcc_denom)
    
    return pcc

<<<<<<< Updated upstream

=======
@jit(nopython=True)
def pcc_dist(a,b):
    '''
    Calculate pearson's corr coef of two matrices, ignoring nans.
    '''
    ind = (a>0) & (b>0)

    a = a[ind]
    b = b[ind]
    
    a_m=np.nanmean(a)
    b_m=np.nanmean(b)
    
    pcc_num=np.nansum((a-a_m)*(b-b_m))
    pcc_denom=np.sqrt(np.nansum((a-a_m)**2))*np.sqrt(np.nansum((b-b_m)**2))
    pcc=np.divide(pcc_num,pcc_denom)
    
    return np.sqrt(1-pcc)

@jit(nopython=True)
def pcc_match(a,b):
    '''
    Calculate pearson's corr coef of two matrices, ignoring nans.
    '''
    ind = (a>0) & (b>0)

    a = a[ind]
    b = b[ind]
    
    a_m=np.nanmean(a)
    b_m=np.nanmean(b)
    
    pcc_num=np.nansum((a-a_m)*(b-b_m))
    pcc_denom=np.sqrt(np.nansum((a-a_m)**2))*np.sqrt(np.nansum((b-b_m)**2))
    pcc=np.divide(pcc_num,pcc_denom)
    
    return pcc

@jit(nopython=True)
def contact_dist(a,b):
    '''
    Calculate pearson's corr coef of two matrices, ignoring nans.
    '''
    ind = (a>0) & (b>0)

    a = a[ind]
    b = b[ind]
    
    score = np.sum((a < 120) & (b < 120))
    
    return 1-score/ind.size

@njit
>>>>>>> Stashed changes
def radius_of_gyration(point_set):
    '''
    Calculate ROG: R = sqrt(1/N * sum((r_k - r_mean)^2) for k points in structure.) 
    Source: https://en.wikipedia.org/wiki/Radius_of_gyration
    '''

    #Only include points passing QC:
    #qc_idx = point_set[:,3] != 0
    #point_set_qc = point_set[qc_idx, 0:3]

    points_mean=numba_mean_axis0(point_set)
    rog = np.sqrt(1/point_set.shape[0] * np.sum((point_set - points_mean)**2))
    return rog

def elongation(point_set):
    '''
    Elongation in this case is defined as the ratio between the two primary
    eigenvalues of the point set.
    '''

    #Only include points passing QC:
    #qc_idx = point_set[:,3] != 0
    #point_set_qc = point_set[qc_idx, 0:3]

    #Center points to 0-mean
    points_centered = point_set - np.mean(point_set, axis=0)
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
    '''Calculate controur length of a trace.

    Args:
        point_set ([4d np array]): coordinate points vector with QC

    Returns:
        [float]: the contour length (sum of next point distances)
    '''
    #Only include points passing QC:
    #qc_idx = point_set[:,3] != 0
    #point_set_qc = point_set[qc_idx, 0:3]
    dist = 0
    for i in range(point_set.shape[0]-1):
        dist += np.linalg.norm(point_set[i+1]-point_set[i])
    return dist

def trace_metrics(traces, use_interp = False, diagonal = 0, contact_cutoff = 150, only_small_loops = False):
    points_nan = points_from_traces_nan(traces)
    points_qc = points_from_traces_qc_filt(traces)
    #ind_u = np.triu_indices(points_nan[0].shape[0], k=2)
    ind_l = np.tril_indices(points_nan[0].shape[0], k=0)
    trace_ids = traces.trace_id.unique()
    metrics = []
    loop_metrics = []
    for i in tqdm.tqdm(range(len(points_nan))):
        if use_interp:
            point_set = np.column_stack(spline_interp([points_qc[i][:,0],points_qc[i][:,1],points_qc[i][:,2]]))
        else:
            point_set = points_nan[i]
        dists = cdist(point_set, point_set)
        ind_l = np.tril_indices(point_set.shape[0], k=diagonal)
        dists_diag = dists.copy()
        dists_diag[ind_l] = np.nan
        contacts = dists_diag < contact_cutoff

        contact_coords = np.argwhere(contacts)
        #print(contact_coords)
        stacked_loops = []
        for c in contact_coords:
            first_anchor = contact_coords[np.argwhere(contact_coords[:,0] == c[0]), :]
            for a in first_anchor:
                try:
                    second_anchor = np.all(contact_coords == [a[0,1],c[1]], axis=1)
                    if np.any(second_anchor):
                        stacked_loops.append(c)
                except IndexError:
                    continue
        if len(stacked_loops) > 0:
            small_loops = contact_coords[~(contact_coords[:, None] == stacked_loops).all(-1).any(-1)]
            #print(small_loops)
        else:
            small_loops = []

        #print(contact_coords)
        #print(stacked_loops)
        n_contacts = np.sum(contacts)
        trace_elongation = elongation(points_qc[i])
        rog = radius_of_gyration(points_qc[i])
        contour = contour_length(points_qc[i])
        #hull_volume = ConvexHull(points_qc[i]).volume
        nn_dist = np.nanmean(np.diagonal(dists, 1))
        if len(contact_coords) == 0:
            freq_nested = np.nan
        else:
            freq_nested = len(stacked_loops)/len(contact_coords)
        #print(small_loops)
        if only_small_loops:
            all_loops = small_loops
        else:
            all_loops = contact_coords
        loop_contours = []
        for j, loop_coords in enumerate(all_loops):
            loop_points = point_set[loop_coords[0]:loop_coords[1]+1]
            loop_points = loop_points[~np.isnan(loop_points).any(axis=1), :]
            loop_contour = contour_length(loop_points)
            #print(loop_coords, stacked_loops)
            try:
                stacked = np.any(np.all(np.array(loop_coords) == np.array(stacked_loops), axis=(1)))
            except np.AxisError:
                stacked = False
            loop_metrics.append([trace_ids[i], j, loop_coords[0], loop_coords[1], loop_contour, stacked])
            loop_contours.append(loop_contour)
        av_loop_size = np.mean(loop_contours)

        interp_point_set = np.column_stack(spline_interp([points_qc[i][:,0],points_qc[i][:,1],points_qc[i][:,2]]))

        interp_contour = contour_length(interp_point_set)
        interp_rog = radius_of_gyration(interp_point_set)

        metrics.append([trace_ids[i], n_contacts, trace_elongation, rog, contour, av_loop_size, nn_dist, interp_contour, interp_rog, freq_nested])
    metrics = pd.DataFrame(metrics, columns=['trace_id', 'n_contacts','elongation', 'rog', 'contour', 'av_loop_size', 'av_nn_dist', 'interp_contour', 'interp_rog', 'freq_nested'])
    loop_metrics = pd.DataFrame(loop_metrics, columns = ['trace_id', 'loop_id', 'loop_coords_0', 'loop_coords_1','contour', 'stacked'])
    loop_metrics['loop_coords_dist'] = loop_metrics['loop_coords_1']-loop_metrics['loop_coords_0']
    loop_metrics['loop_coords'] = loop_metrics['loop_coords_0'].astype(str).str.zfill(2)+'_'+loop_metrics['loop_coords_1'].astype(str).str.zfill(2)
    return metrics, loop_metrics

def looping_distribution_from_simulated_loops(loop_pos, loop_anchors):
    '''Calculate types of loops from simulated loop positions compared to given anchors.
    Types are:
    1: Overlap with both anchors
    2: Overlap with one anchor and other base inside second anchor
    3: Overlap with one anchor and other base outside second anchor
    4: Small loop inside both anchors
    5: Large loops outside both anchors
    6: No loop overlapping with loop anchor region

    Args:
        loop_pos (nd.array): np.array of shape N loops X M SMCs X 2 anchors
        loop_anchors (tuple): iterable with the two loop anchors to measure compared to
    Returns:
        loop_distribution (list): The fraction of cells with the given type of loop.
        res (ndarray): The class of loop formed by per cell (rows) by all SMCs 
    '''
    res = np.zeros((loop_pos.shape[:2]))
    for i, cell in enumerate(loop_pos):
        for j,l in enumerate(cell):
            #print(l)
            if (l[0] == loop_anchors[0]) and (l[1] == loop_anchors[1]): #Overlap with both anchors
                res[i,j] = 1
            elif (l[0] == loop_anchors[0]) and (l[1] < loop_anchors[1]) or (l[0] > loop_anchors[0]) and (l[1] == loop_anchors[1]): #Overlap with one anchor and small loop
                res[i,j] = 2
            elif (l[0] == loop_anchors[0]) and (l[1] > loop_anchors[1]) or (l[0] < loop_anchors[0]) and (l[1] == loop_anchors[1]): #Overlap with one anchor and large loop
                res[i,j] = 3
            elif (l[0] > loop_anchors[0]) and (l[1] < loop_anchors[1]): #Small loop inside anchors
                res[i,j] = 4
            elif (l[0] < loop_anchors[0]) and (l[1] > loop_anchors[1]): #Large loops outside anchors
                res[i,j] = 5
            else:
                res[i,j] = 0 #No loop
    loop_distribution = [np.sum(np.all(res == 0, axis = 1))/len(res)] + [np.sum(np.any(res == i, axis = 1))/len(res) for i in range(1,6)]
    return loop_distribution, res

def fit_plane_SVD(points):
    '''
    From:
    https://stackoverflow.com/questions/15959411/fit-points-to-a-plane-algorithms-how-to-iterpret-results

    Args:
        XYZ ([type]): [description]

    Returns:
        [type]: [description]
    '''

    [rows,cols] = points.shape
    # Set up constraint equations of the form  AB = 0,
    # where B is a column vector of the plane coefficients
    # in the form b(1)*X + b(2)*Y +b(3)*Z + b(4) = 0.
    p = (np.ones((rows,1)))
    AB = np.hstack([points,p])
    [u, d, v] = np.linalg.svd(AB,0)        
    B = v[3,:];                    # Solution is last column of v.
    nn = np.linalg.norm(B[0:3])
    B = B / nn
    return B[0:3]

<<<<<<< Updated upstream
def plot_heatmap(traces, trace_id='all', zmin=0, zmax=600, cmap = 'RdBu'):
=======
def plot_heatmap(traces, trace_ids=None, ax=None, zmin=0, zmax=600, cmap = 'RdBu', crop=True, **kwargs):
>>>>>>> Stashed changes
    '''Helper function to make heatmaps of sets of traces.

    Args:
        traces ([DataFrame]): trace DataFrame
        trace_id (str or int or list, optional): trace_ids in dataframe to use. Defaults to 'all'.
        zmax (int, optional): Max value of heatmap. Defaults to 600.
        cmap (str, optional): Color scale for heatmap. Defaults to 'rdbu'.

    Returns:
        pwds_crop: Numpy array, Data underlying heatmap.
        fig: Figure object of the heatmap.
    '''
    if trace_ids is None:
            pwds = pwd_calc(traces)
            pwds_mean = np.nanmedian(pwds, axis=0)
    else:
        if type(trace_ids) == int:
            trace_ids = [trace_ids]
        pwds = pwd_calc(traces[traces['trace_id'].isin(trace_ids)])
        pwds_mean = np.nanmedian(pwds, axis=0)

    if crop:
        nan_rows = ~np.all(np.isnan(pwds_mean), axis=0)
        nan_cols = ~np.all(np.isnan(pwds_mean), axis=1)
        pwds_mean = pwds_mean[nan_rows,:]
        pwds_mean = pwds_mean[:,nan_cols]
    print('Number of traces in heatmap: ', pwds.shape[0])
<<<<<<< Updated upstream
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(pwds_crop, vmin=zmin, vmax=zmax, cmap=cmap)
    #fig.show()
    return pwds_crop, ax
    
=======
    ax = ax or plt.gca()
    _cmap = plt.get_cmap(cmap).copy()
    _cmap.set_bad('lightgrey')
    plot = ax.imshow(pwds_mean, vmin=zmin, vmax=zmax, cmap=_cmap, **kwargs)
    ax.axis('off')
    #fig.show()
    return pwds_mean, plot

def plot_contacts(traces, trace_ids=None, cutoff=150, ax=None, zmin=0, zmax=1, cmap = 'RdBu_r', crop=True, **kwargs):
    '''Helper function to make heatmaps of sets of traces.

    Args:
        traces ([DataFrame]): trace DataFrame
        trace_id (str or int or list, optional): trace_ids in dataframe to use. Defaults to 'all'.
        zmax (int, optional): Max value of heatmap. Defaults to 600.
        cmap (str, optional): Color scale for heatmap. Defaults to 'rdbu'.

    Returns:
        pwds_crop: Numpy array, Data underlying heatmap.
        fig: Figure object of the heatmap.
    '''
    if trace_ids is None:
        pwds = pwd_calc(traces)
        dist = np.nansum(pwds<cutoff, axis=0)/np.nansum(pwds>0, axis=0)

    else:
        if type(trace_ids) == int:
            trace_ids = [trace_ids]
        pwds = pwd_calc(traces[traces['trace_id'].isin(trace_ids)])
        dist = np.nansum(pwds<cutoff, axis=0)/np.nansum(pwds>0, axis=0)


    if crop:
        nan_rows = ~np.all(np.isnan(dist), axis=0)
        nan_cols = ~np.all(np.isnan(dist), axis=1)
        dist = dist[nan_rows,:]
        dist = dist[:,nan_cols]
    dist = np.nan_to_num(dist, posinf=1)
    print('Number of traces in heatmap: ', pwds.shape[0])
    ax = ax or plt.gca()
    _cmap = plt.get_cmap(cmap).copy()
    _cmap.set_bad('lightgrey')
    plot = ax.imshow(dist, vmin=zmin, vmax=zmax, cmap=_cmap, **kwargs)
    ax.axis('off')
    #fig.show()
    return dist, plot
    

def plot_n_positions(traces, trace_id='all', ax=None, zmin=0, zmax=1, cmap = 'viridis', **kwargs):
    '''Helper function to make heatmaps of sets of traces.

    Args:
        traces ([DataFrame]): trace DataFrame
        trace_id (str or int or list, optional): trace_ids in dataframe to use. Defaults to 'all'.
        zmax (int, optional): Max value of heatmap. Defaults to 600.
        cmap (str, optional): Color scale for heatmap. Defaults to 'rdbu'.

    Returns:
        pwds_crop: Numpy array, Data underlying heatmap.
        fig: Figure object of the heatmap.
    '''
    if trace_id != 'all':
        if type(trace_id) == int:
            trace_id = [trace_id]
        pwds = pwd_calc(traces[traces['trace_id'].isin(trace_id)])
        dist = np.nansum(pwds>0, axis=0)
    else:
        pwds = pwd_calc(traces)
        dist = np.nansum(pwds>0, axis=0)

    nan_rows = ~np.all(np.isnan(dist), axis=0)
    nan_cols = ~np.all(np.isnan(dist), axis=1)
    pwds_crop = dist[nan_rows,:]
    pwds_crop = pwds_crop[:,nan_cols]
    pwds_crop = np.nan_to_num(pwds_crop, posinf=1)
    print('Number of traces in heatmap: ', pwds.shape[0])
    ax = ax or plt.gca()
    _cmap = plt.get_cmap(cmap).copy()
    _cmap.set_bad('lightgrey')
    plot = ax.imshow(pwds_crop, vmin=zmin, vmax=zmax, cmap=_cmap, **kwargs)
    ax.axis('off')
    #fig.show()
    return pwds_crop, plot
>>>>>>> Stashed changes

def plot_traces(traces, trace_id, split=False):
    '''
    Helper function for plotting one or several traces in one figure.
    Also plots spline interpolation between points for visualization.
    
    Parameters
    ----------
    traces : pd DataFrame with trace data.
    trace_id : Int or list of ints with trace_id of traces to plot.     

    Returns
    ----------
    Fig object for saving or further manipulation.
    '''

    if type(trace_id) == int:
        trace_id=[trace_id]
    df=traces[traces['trace_id'].isin(trace_id)]
    df['keys'] = df['frame_name'].astype(str).str[0]
    labels=list(df['frame_name'])
    print(labels)

    fig = px.scatter_3d(df, x='x', y='y', z='z', 
                symbol='keys',
                color='frame_name', 
                color_discrete_sequence = px.colors.sequential.Inferno,
                labels={'frame_name' : 'Exchange', 'keys':'Group'})

    for i in trace_id:
        for key in list(df['keys'].unique()):
            df_i = df[(df['trace_id'] == i) & (df['keys'] == key)]
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

    fig.update_layout(template='plotly_white', 
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
    trace_id : Int or list of ints with trace_id of traces to plot. 

    Returns
    ----------
    Fig object for saving or further manipulation.

    '''


    all_points = points_from_traces(traces, idx)
    template = all_points.pop(0)
    template = center_points_qc(template)
    all_points_aligned = [template]+[rigid_transform_3D(offset, template) for 
                          offset in all_points]
    scatters = []
    cmap = px.colors.sequential.Inferno
    for point_id, point_set in enumerate(all_points_aligned):
        idx=np.arange(point_set.shape[0])
        qc_idx = point_set[:,3] != 0
        idx=idx[qc_idx]
        labels=['E'+str(i) for i in idx]
        cmap_points = [cmap[(i)%len(cmap)] for i in idx]
        z,y,x = point_set[qc_idx, 0], point_set[qc_idx, 1], point_set[qc_idx, 2]
        print(z,y,x)
        z_f, y_f, x_f=spline_interp([z,y,x])
        scatters.append(go.Scatter3d(x=x, y=y, z=z, 
                                     mode='markers  ', 
                                     marker_color=cmap_points,
                                     marker_size=9,
                                     opacity=1,
                                     name='Trace '+str(point_id)))
        scatters.append(go.Scatter3d(x=x_f, 
                            y=y_f, 
                            z=z_f,
                            mode ='lines',
                            name='Trace '+str(point_id),
                            line=dict(color=px.colors.qualitative.Plotly[point_id],
                                        width=6)))
    fig = go.Figure(data=scatters)
    fig.update_layout(
    font=dict(size=18),
    template='plotly_white', 
    showlegend= True,
    height=600,
    scene =
    dict(
    aspectmode='data',
    xaxis=dict(showgrid = True, nticks=5),
    yaxis=dict(showgrid = True, nticks=5),
    zaxis=dict(showgrid = True, nticks=5),
    xaxis_title='x [nm]',
    yaxis_title='y [nm]',
    zaxis_title='z [nm]',
    ))

    #iplot(fig)
    return fig
    
def plot_gpa_output(aligned_points, mean_points, cluster_members=None, cluster_id=0, mean_color=None):
    '''
    Helper function for plotting the results of a GPA analysis.
    
    Parameters
    ----------
    aligned_points : List of aligned point sets
    mean_points: Nx4 array of mean points
    cluster_members: List of trace_ids of the aligned points, typically from trace_clustering output

    Returns
    ----------
    Fig object for saving or further manipulation.

    '''
    if cluster_members is None:
        cluster_members = list(range(len(aligned_points)))
    scatters = []
    cmap = px.colors.sequential.Inferno
    cmap_line = px.colors.qualitative.Plotly
    if mean_color is None:
        mean_color = cmap_line[cluster_id-1]
    for point_id, point_set in enumerate(aligned_points):
        idx=np.arange(point_set.shape[0])
        qc_idx = point_set[:,3] != 0
        idx=idx[qc_idx]
        labels=['E'+str(i) for i in idx]
        
        cmap_points = [cmap[(i%len(cmap))] for i in idx]
        z,y,x = point_set[qc_idx, 0], point_set[qc_idx, 1], point_set[qc_idx, 2]
        z_f, y_f, x_f=spline_interp([z,y,x])
        scatters.append(go.Scatter3d(x=x, 
                        y=y, 
                        z=z,
                        mode ='lines',
                        showlegend=False,
                        line=dict(color='rgba(50, 50, 50, 0.0)',
                                    width=3)))
        scatters.append(go.Scatter3d(x=x, y=y, z=z, 
                                     mode='markers', 
                                     marker_color=cmap_points,
                                     marker_size=3,
                                     opacity=0.2,
                                     name='Trace '+str(cluster_members[point_id]),
                                     ))

    mean_idx=np.arange(mean_points.shape[0])
    mean_qc = mean_points[:,3] != 0
    mean_idx=mean_idx[mean_qc]
    print(mean_idx)
    #mean_labels=['E'+str(i) for i in mean_idx]
    mean_cmap = [cmap[i%10] for i in mean_idx]
    z_m, y_m, x_m = mean_points[mean_idx, 0], mean_points[mean_idx, 1], mean_points[mean_idx, 2]
    z_fm, y_fm, x_fm=spline_interp([z_m,y_m,x_m])
    scatters.append(go.Scatter3d(x=x_m, y=y_m, z=z_m, 
                                            mode='markers', 
                                            marker_color=mean_cmap,
                                            marker_size=8
                                            ))
    
    scatters.append(go.Scatter3d(x=x_fm, 
                    y=y_fm, 
                    z=z_fm,
                    mode ='lines',
                    showlegend=False,
                    name='Mean',
                    line=dict(color=mean_color,
                                width=6)))
    

    fig = go.Figure(data=scatters)
    fig.update_layout(
    font=dict(size=16),
    template='plotly_white', 
    showlegend= True,
    height=600,
    scene =
    dict(
    xaxis=dict(showgrid = False, nticks=0),
    yaxis=dict(showgrid = False, nticks=0),
    zaxis=dict(showgrid = False, nticks=0),
    xaxis_title='X [nm]',
    yaxis_title='Y [nm]',
    zaxis_title='Z [nm]'
    ))
    #iplot(fig)
    return fig

def plot_gpa_output_std(aligned_points, mean_points, mean_color=None):
    '''
    Helper function for plotting the results of a GPA analysis.
    
    Parameters
    ----------
    aligned_points : List of aligned point sets
    mean_points: Nx4 array of mean points
    cluster_members: List of trace_ids of the aligned points, typically from trace_clustering output

    Returns
    ----------
    Fig object for saving or further manipulation.

    '''
    
    scatters = []
    cmap = px.colors.sequential.Inferno
    cmap_line = px.colors.qualitative.Plotly
    if mean_color is None:
        mean_color = 'pink'

    mean_idx=np.arange(mean_points.shape[0])
    mean_qc = mean_points[:,3] != 0
    mean_idx=mean_idx[mean_qc]
    print(mean_idx)
    #mean_labels=['E'+str(i) for i in mean_idx]
    mean_cmap = [cmap[i%10] for i in mean_idx]

    std_dist = np.nanstd(np.sqrt(np.nansum((np.stack(aligned_points) - mean_points)[:,:,:3]**2, axis=2)), axis=0)
    z_m, y_m, x_m = mean_points[mean_idx, 0], mean_points[mean_idx, 1], mean_points[mean_idx, 2]
    z_fm, y_fm, x_fm=spline_interp([z_m,y_m,x_m])
    scatters.append(go.Scatter3d(x=x_m, y=y_m, z=z_m, 
                                            mode='markers', 
                                            marker_color=mean_cmap,
                                            marker_size = std_dist,
                                            opacity = 0.5
                                            ))
    scatters.append(go.Scatter3d(x=x_m, y=y_m, z=z_m, 
                                        mode='markers', 
                                        marker_color=mean_cmap,
                                        marker_size=10
                                        ))
    
    scatters.append(go.Scatter3d(x=x_fm, 
                    y=y_fm, 
                    z=z_fm,
                    mode ='lines',
                    showlegend=False,
                    name='Mean',
                    line=dict(color=mean_color,
                                width=8)))
    

    fig = go.Figure(data=scatters)
    fig.update_layout(
    font=dict(size=16),
    template='plotly_white', 
    showlegend= True,
    height=800,
    width=1000,
    scene =
    dict(
    xaxis=dict(showgrid = False, nticks=3, visible = False),
    yaxis=dict(showgrid = False, nticks=3, visible = False),
    zaxis=dict(showgrid = False, nticks=3, visible = False),
    xaxis_title='',
    yaxis_title='',
    zaxis_title=''
    ))
    #iplot(fig)
    return fig
    
def plot_multi_points(list_of_points, names = None, line_color='#1f77b4'):
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
    cmap = px.colors.sequential.Inferno
    cmap_line = px.colors.qualitative.Plotly
    for point_id, point_set in enumerate(list_of_points):
        idx=np.arange(point_set.shape[0])
        qc_idx = point_set[:,3] != 0
        idx=idx[qc_idx]
        labels=['E'+str(i) for i in idx]
        
        cmap_points = [cmap[(i)%len(cmap)] for i in idx]
        if names is not None:
            name = names[point_id]
        else:
            name = 'Cluster '+str(point_id+1)
        p = point_set[qc_idx,:3]
        p = p-np.mean(p, axis=0)
        p = p + np.abs(np.min(p, axis=0))
        z,y,x = p[:,0], p[:,1], p[:,2]
        
        z_f, y_f, x_f=spline_interp([z,y,x])

        scatters.append(go.Scatter3d(x=x, y=y, z=z, 
                                     mode='markers', 
                                     marker_color=cmap_points,
                                     marker_size=15,
                                     opacity=1,
                                     name=name,
                                     showlegend=True,))
                                     #line=dict(color=cmap_line[point_id],
                                               #width=4)))
        scatters.append(go.Scatter3d(x=x_f, 
                        y=y_f, 
                        z=z_f,
                        mode ='lines',
                        showlegend=True,
                        line=dict(color=line_color,#point_id],
                                    width=12)))
    
    fig = go.Figure(data=scatters)
    fig.update_layout(
    font=dict(size=16),
    template='plotly_white', 
    showlegend= True,
    height=800,
    width=1000,
    scene =
    dict(
    xaxis=dict(showgrid = False, nticks=3),
    yaxis=dict(showgrid = False, nticks=3),
    zaxis=dict(showgrid = False, nticks=3),
    xaxis_title='',
    yaxis_title='',
    zaxis_title=''
    ))
    #iplot(fig)
    return fig

def plot_2d_proj(points, std_points=None, line_color='#1f77b4', plane='best', limits=(-450,450)):
    '''[summary]

    Args:
        points ([type]): Nx4 ndarray of 3D points with QC.
        std_points ([type], optional): N-vector of STD of points. Defaults to None.
        line_color (str, optional): Provide RGB value or hex of spline fit line. Defaults to '#1f77b4'.
        plane (str, optional): Choose the desired projection plane, e.g. 'xy', 'yz'. Defaults to 'best',
                                least squares best fit plane for the 3D data.

    Returns:
        [type]: Figure object of plotted data.
    '''


    qc = points[:,3] == 1

    if plane != 'best':
        axis_map = {'z':0, 'y':1, 'x':2}
        axis_x = axis_map[plane[0]]
        axis_y = axis_map[plane[1]]
        x = points[qc,axis_x]
        y = points[qc,axis_y]
        x = x-np.mean(x)
        y = y-np.mean(y)
        xf, yf = spline_interp(np.array([x,y]), 500)
    elif plane == 'xz':
        x = points[qc,2]
        y = points[qc,1]
        x = x-np.mean(x)
        y = y-np.mean(y)
        xf, yf = spline_interp(np.array([x,y]), 500)
    
    else:
        n = fit_plane_SVD(points[qc,:3])
        v1 = np.cross(np.array([0,0,1]), n) # Find a random orthogonal vector to n (in the plane)
        v1 = v1/np.linalg.norm(v1)  # normalize it
        v2 = np.cross(n, v1) # Find the third orthogonal vector

        x = np.dot(points[qc,:3],v1)
        y = np.dot(points[qc,:3],v2)
        x = x-np.mean(x)
        y = y-np.mean(y)
        xf, yf = spline_interp(np.array([x,y]), 500)
    positions = np.array(range(points.shape[0]))
    positions = list(positions[qc])
<<<<<<< Updated upstream

    fig = plt.figure()
    sns.scatterplot(y,x, hue=positions, palette='inferno', legend=None, alpha=1, s=100, linewidth=0)
    if std_points is not None:
        sns.scatterplot(y,x, hue=positions, palette='inferno', legend=None, alpha=0.3, sizes=list((std_points/2)**2), size=positions, linewidth=0)
    plt.plot(yf,xf, zorder=-10, color=line_color, linewidth=3)
=======
    ax = ax or plt.gca()
    marker_size =  int(50*np.sqrt(450/limits[1]))
    sns.scatterplot(x=y,y=x, hue=positions, clip_on=False, palette='gnuplot', legend=None, alpha=1, s=marker_size, edgecolor=None,  zorder=20, ax=ax)
    if std_points is not None:
        sns.scatterplot(x=y,y=x, hue=positions, palette='inferno', legend=None, alpha=0.3, sizes=list((std_points/2)**2), size=positions, linewidth=0, ax=ax)
    ax.plot(yf,xf, zorder=-10, color=line_color, linewidth=2)

    ax.set_ylim(limits)
    ax.set_xlim(limits)
    ax.set_aspect('equal')
    ax.axis('off')

    if scale:
        from matplotlib_scalebar.scalebar import ScaleBar
        scalebar = ScaleBar(1, "nm", fixed_value=100)
        ax.add_artist(scalebar)

    return ax

def plot_2d_proj_kde(mean_points, aligned_points, ax=None, line_color='#1f77b4', limits=(-450,450), scale=True, subselect = False, kde=True):
    from matplotlib import cm

    if subselect:
       palette = [cm.get_cmap('gnuplot', mean_points.shape[0])(i) for i in range(mean_points.shape[0])][subselect[0]:subselect[1]]
       mean_points = mean_points[subselect[0]:subselect[1]]
       aligned_points = [points[subselect[0]:subselect[1]] for points in aligned_points]
    else:
       palette = 'gnuplot'
>>>>>>> Stashed changes

    plt.ylim(limits)
    plt.xlim(limits)
    plt.gca().set_aspect('equal')
    plt.axis('off')
    return fig

def plot_2d_proj_kde(mean_points, aligned_points, line_color='#1f77b4', limits=(-450,450)):
    
    qc = mean_points[:,3] == 1
    n = fit_plane_SVD(mean_points[qc,:3])
    v1 = np.cross(np.array([0,0,1]), n) # Find a random orthogonal vector to n (in the plane)
    v1 = v1/np.linalg.norm(v1)  # normalize it
    v2 = np.cross(n, v1) # Find the third orthogonal vector

    positions = np.array(range(mean_points.shape[0]))
    
    x_m = np.dot(mean_points[qc,:3],v1)
    y_m = np.dot(mean_points[qc,:3],v2)
    x_c = np.mean(x_m)
    y_c = np.mean(y_m)
    x_m = x_m-x_c
    y_m = y_m-y_c

    xf, yf = spline_interp(np.array([x_m,y_m]), 500)

    x_all = []
    y_all = []   
    pos_all = [] 

    for points in aligned_points:
        qc = points[:,3] == 1
        qc_pos = positions[qc]
        x = np.dot(points[qc,:3],v1)
        y = np.dot(points[qc,:3],v2)
        x = x-x_c
        y = y-y_c
        x_all.append(x)
        y_all.append(y)
        pos_all.append(qc_pos)

    x_all = np.hstack(x_all)
    y_all = np.hstack(y_all)
    pos_all = np.hstack(pos_all)

    data = pd.DataFrame(np.array([x_all, y_all, pos_all]).T, columns=['x', 'y', 'pos'])

    fig = plt.figure(figsize=(8,8))

    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    cmap_inferno = cm.get_cmap('inferno', 10).colors

#    for p in positions:
#        sel_data = data.query('pos == @p')
#        N = 256
#        vals = np.ones((N, 4))
#        vals[:, 0] = np.linspace(cmap_inferno[p][0], cmap_inferno[p][0], N)
#        vals[:, 1] = np.linspace(cmap_inferno[p][1], cmap_inferno[p][1], N)
#        vals[:, 2] = np.linspace(cmap_inferno[p][2], cmap_inferno[p][2], N)
#        vals[:, 3] = np.linspace(0, 1, N)
#        newcmp = ListedColormap(vals)
<<<<<<< Updated upstream
=======
    ax = ax or plt.gca()
    if kde:
        sns.kdeplot(ax=ax, data=data, x='y', y='x', hue='pos', palette=palette, linewidths=0.05, common_norm=False, fill=False, legend=None, levels = 25, thresh=0.75, alpha=0.4)
    #sns.scatterplot(data=data, x='y', y='x', hue='pos', palette='inferno', s = 6, alpha = 0.3, legend=None)
    ax.plot(yf,xf, zorder=10, color=line_color, linewidth=3, clip_on=False)
    #print(y_m, x_m)
    color_positions = np.array(range(y_m.shape[0]))
    marker_size =  int(50*np.sqrt(450/limits[1]))
    sns.scatterplot(x=y_m,y=x_m, ax=ax, hue=color_positions, clip_on=False, palette=palette, legend=None, alpha=1, s=marker_size, edgecolor=None,  zorder=20)
>>>>>>> Stashed changes

    sns.kdeplot(data=data, x='y', y='x', hue='pos', palette='inferno', linewidths=0.3, common_norm=False, fill=False, legend=None, levels = 25, thresh=0.75, alpha=0.2)
    #sns.scatterplot(data=data, x='y', y='x', hue='pos', palette='inferno', s = 6, alpha = 0.3, legend=None)

    plt.plot(yf,xf, zorder=10, color=line_color, linewidth=3, clip_on=False)
    sns.scatterplot(y_m,x_m, hue=positions, clip_on=False, palette='inferno', legend=None, alpha=1, s=100, edgecolor=None,  zorder=20)

    plt.ylim(limits)
    plt.xlim(limits)
    plt.gca().set_aspect('equal')
    plt.axis('off')
    return fig

def plot_single_trace_grid(aligned_points, mean_points, max_n = 50, proj_plane = None, show_mean = False, line_color='#1f77b4'):
    max_n = min(len(aligned_points), max_n)
    if proj_plane == 'mean':
        qc = mean_points[:,3] == 1
        n = fit_plane_SVD(mean_points[qc,:3])
        v1 = np.cross(np.array([0,0,1]), n) # Find a random orthogonal vector to n (in the plane)
        v1 = v1/np.linalg.norm(v1)  # normalize it
        v2 = np.cross(n, v1) # Find the third orthogonal vector

    n_rows = int(np.sqrt(max_n))
    fig, axs = plt.subplots(n_rows, n_rows, figsize=(10,9), sharex=False, sharey=False)
    axs = axs.ravel()

    if show_mean:
        aligned_points = [mean_points] + aligned_points[:n_rows**2-1]
    else:
        aligned_points = aligned_points[:n_rows**2]

    for i, points in enumerate(aligned_points):
        qc = points[:,3] == 1
        if proj_plane is None:
            n = fit_plane_SVD(points[qc,:3])
            v1 = np.cross(np.array([0,0,1]), n)
            v1 = v1/np.linalg.norm(v1)  # normalize it
            v2 = np.cross(n, v1)
        x = np.dot(points[qc,:3],v1)
        y = np.dot(points[qc,:3],v2)
        x = x-np.mean(x)
        y = y-np.mean(y)
        xf, yf = spline_interp(np.array([x,y]), 500)
        positions = np.array([0,1,2,3,4,5,6,7,8,9])
        positions = list(positions[qc])
        sns.scatterplot(y,x, ax=axs[i], hue=positions, palette='inferno', legend=None, alpha=0.8, s=30, edgecolor=None, clip_on=False)#sizes=sizes, size=positions)
        axs[i].plot(yf,xf, zorder=-10, color=line_color, clip_on=False)
        axs[i].set_ylim(-450,450)
        axs[i].set_xlim(-450,450)
        axs[i].axis('off')
        axs[i].set_aspect('equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    return fig

def plot_mds(cluster_df, pos, cluster_method = 'dendro'):
    fig = plt.figure(figsize=(15,15))
    sns.set(font_scale=2)
    sns.set_palette(sns.color_palette('tab10')[:])
    sns.set_style('ticks')
    hue_order = sorted(cluster_df[cluster_method].unique().astype(str))
    sns.kdeplot(x=pos[:,0], y=pos[:,1], hue=cluster_df[cluster_method].astype(str),  levels=20, thresh=.1, alpha=0.3, hue_order=hue_order)
    sns.scatterplot(x=pos[:,0], y=pos[:,1], hue=cluster_df[cluster_method].astype(str), s=200, alpha=1, hue_order=hue_order)
    
    return fig

def plot_aligned_traces_animated(aligned_points):
    frames = []
    cmap = px.colors.sequential.Inferno
    for i, points in enumerate(aligned_points):
        idx=np.arange(points.shape[0])
        qc_idx = points[:,3] != 0
        idx=idx[qc_idx]
        cmap_points = [cmap[(i-1)%len(cmap)] for i in idx]
        z,y,x = points[qc_idx,0], points[qc_idx,1], points[qc_idx,2]
        z_f, y_f, x_f=spline_interp([z,y,x])
        frames.append(go.Frame(data=[go.Scatter3d(
                                    x=x, 
                                    y=y,
                                    z=z,
                                    mode='markers', 
                                    name= 'trace_'+str(i),
                                    marker_color=cmap_points,
                                    marker_size=6,),
                                    go.Scatter3d(x=x_f, 
                                    y=y_f, 
                                    z=z_f,
                                    mode ='lines')]))

    fig = go.Figure(frames = frames)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, name='last', marker_color=cmap_points,
                                    marker_size=6,mode='markers'))
    fig.add_trace(go.Scatter3d(x=x_f, y=y_f, z=z_f, mode ='lines'))

    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]
    # Layout
    fig.update_layout(
            title='Traces',
            width=800,
            height=800,
            updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(500)],
                            "label": "&#9654;", # play symbol
                            "method": "animate"},
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate"},
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=sliders
    )
    print(x)
    fig.show(renderer='notebook')

def animate_trace(clust_aligned, t_interval=200, n_points=500):
    import ipyvolume as ipv
    x = []
    y = []
    z = []
    xs = []
    ys = []
    zs = []

    for i, points in enumerate(clust_aligned):
        qc_idx = points[:,3] != 0
        if np.sum(qc_idx) != 0:
            x_ = points[qc_idx,2]
            x_ = x_-np.mean(x_)
            y_ = points[qc_idx,1]
            y_ = y_-np.mean(y_)
            z_ = points[qc_idx,0]
            z_ = z_-np.mean(z_)
            x.append(x_)
            y.append(y_)
            z.append(z_)
            xs_,ys_,zs_ = spline_interp([x_,y_,z_], n_points=n_points)
            xs.append(xs_)
            ys.append(ys_)
            zs.append(zs_)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    #inferno_colors, _ = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Inferno)
    #colorscale = plotly.colors.make_colorscale(inferno_colors)
    #colors = np.array([get_continuous_color(colorscale, intermed=i/x.shape[1]) for i in range(0,x.shape[1],1)]).astype(int)/255
    #selected = np.array([1])
    fig = ipv.figure()
    ipv.style.axes_off()
    ipv.style.box_off()
    #ipv.style.background_color('grey')
    s1 = ipv.scatter(x,y,z, size = 6,  size_selected=6,  marker='sphere' )#selected=selected,color_selected='lime', color=colors,
    s2 = ipv.scatter(xs,ys,zs, size = 2, color='tab:blue', marker='sphere')
    ipv.animation_control([s1,s2], interval=t_interval) # shows controls for animation controls
    ipv.save(os.getcwd()+os.sep+'ipv.html')
    ipv.show()
    return ipv.gcc()

def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:
    
        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    if intermed <= 0 or len(colorscale) == 1:
        return re.findall('\d*\.?\d+',colorscale[0][1])
    if intermed >= 1:
        return re.findall('\d*\.?\d+',colorscale[-1][1])

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    # noinspection PyUnboundLocalVariable
    rgb = plotly.colors.find_intermediate_color(
        lowcolor=low_color, highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb")
    return [int(float(s)) for s in re.findall('\d*\.?\d+',rgb)]

def animate_trace_mayavi(clust_aligned, n_points=500, duration=10, fps=10, out_dir=os.getcwd()):
    from mayavi import mlab
    mlab.options.offscreen = True
    import moviepy.editor as mpy

    x = []
    y = []
    z = []
    xs = []
    ys = []
    zs = []

    for i, points in enumerate(clust_aligned):
        qc_idx = points[:,3] != 0
        if np.sum(qc_idx) != 0:
            x_ = points[qc_idx,2]
            x_ = x_-np.mean(x_)
            y_ = points[qc_idx,1]
            y_ = y_-np.mean(y_)
            z_ = points[qc_idx,0]
            z_ = z_-np.mean(z_)
            x.append(x_)
            y.append(y_)
            z.append(z_)
            xs_,ys_,zs_ = spline_interp([x_,y_,z_], n_points=n_points)
            xs.append(xs_)
            ys.append(ys_)
            zs.append(zs_)
            
    mlab.figure(1, size=(400, 400), bgcolor=(0, 0, 0))
    mlab.clf()
    l = mlab.plot3d(xs[0]/1000, ys[0]/1000, zs[0]/1000, np.arange(xs[0].shape[0]), tube_radius=0.015, opacity = 0.5, colormap='Spectral')
    l2 = mlab.text(0,0,str(0), width=0.0625)
    # Now animate the data.
    ms = l.mlab_source

    def make_frame(i):
        #mlab.clf()
        i = int(i*10)
        x_new = xs[i]/1000
        y_new = ys[i]/1000
        z_new = zs[i]/1000
        ms.trait_set(x=x_new, y=y_new, z=z_new)
        l2.set(text=str(i))
        mlab.view(azimuth=360/(i+1), distance=2)
        return mlab.screenshot(antialiased=True)

    animation = mpy.VideoClip(make_frame, duration=duration)
    animation.write_gif(r"M:\Kai\tests\test.gif", fps=fps)

def animate_sim_mayavi(point_array, out_path, fps=10, n_points = 100, loops = None):
    from mayavi import mlab
    mlab.options.offscreen = True
    
    import moviepy.editor as mpy

    if point_array.shape[1] < n_points:
        x = np.empty(shape=(point_array.shape[0], n_points))
        y = np.empty(shape=(point_array.shape[0], n_points))
        z =  np.empty(shape=(point_array.shape[0], n_points))
        for i, p in enumerate(point_array):
            x[i], y[i], z[i] = spline_interp([p[:,0], p[:,1], p[:,2]], n_points=n_points)
    else:
        x = point_array[:,:,2]
        y = point_array[:,:,1]
        z = point_array[:,:,0]
    
    mlab.figure(1, size=(400, 400), bgcolor=(1, 1, 1))
    mlab.clf()
    l = mlab.plot3d(x[0], y[0], z[0], np.arange(x.shape[1]), tube_radius=0.005, opacity = 0.5, colormap='Spectral')
    l.scene.light_manager.light_mode = "vtk"
    l2 = mlab.text(0,0,str(0).zfill(3), width=0.0625, color=(0,0,0))
    if loops is not None:
        l3 = mlab.points3d(loops[0,:,2], loops[0,:,1], loops[0,:,0], scale_factor=0.05, opacity= 0.5, color=(0,1,0))
        ms3 = l3.mlab_source
    # Now animate the data.
    ms = l.mlab_source

    def make_frame(i):
        #mlab.clf()
        i = int(i*fps)
        ms.trait_set(x=x[i], y=y[i], z=z[i])
        l2.set(text=str(i).zfill(3))
        if loops is not None:
            ms3.trait_set(x=loops[i,:,2], y=loops[i,:,1], z=loops[i,:,0])
        #mlab.view(azimuth=360/(i+1), distance=1)
        mlab.view(azimuth=45, elevation=60, distance=2, focalpoint=(np.mean(x[i]),np.mean(y[i]),np.mean(z[i])))
        return mlab.screenshot(antialiased=True)

    animation = mpy.VideoClip(make_frame, duration=x.shape[0]//fps)
    animation.write_gif(out_path, fps=fps)
