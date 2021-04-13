import os
import tifffile as tiff
import numpy as np
import glob
import h5py
import scipy.ndimage
import cc3d
import pandas as pd
import skimage.measure
from matplotlib import pyplot as plt



"""
img_path = filelist[0]
def normalise(img_path):
    img = tiff.imread(img_path)
    for time in range(img.shape[0]):
        img[time,:,:,:].max()

# Test case
toplevel_dir = r"F:\ProcessingFolder\20200602_OO_326_TAD-DNA_Fixed-FISHed_tmTit"
input_dir = "02_DC_4D"
img_ext = ".tif"

os.chdir(toplevel_dir)
os.chdir(input_dir)

# = glob.glob('*' + img_ext)
#filelist
"""

#mask_idx = mask
def makeSummaryTable(mask_idx):
    summaryTable = pd.DataFrame(columns=["Label", "Volume", "X_pix", "Y_pix", "Z_pix"])
    for ch in range(mask_idx.shape[1]):
        for fr in range(mask_idx.shape[0]):
            props = skimage.measure.regionprops(mask_idx[fr,ch,:,:,:])
            label, volume, X, Y, Z = [], [], [], [], []
            for prop in props:
                label.append(prop.label)
                volume.append(prop.area)
                z, y, x = prop.centroid
                X.append(x)
                Y.append(y)
                Z.append(z)           
            st = {'Label': label, "Channel": int(ch+1), "Frame": int(fr +1), 'Volume': volume, 'X_pix': X, 'Y_pix': Y, 'Z_pix': Z}
            st = pd.DataFrame(st)
            summaryTable = summaryTable.append(st)
    return summaryTable 

def makeSummaryTable_tzyx(mask_idx):
    summaryTable = pd.DataFrame(columns=["Label", "Volume", "X_pix", "Y_pix", "Z_pix"])
    for fr in range(mask_idx.shape[0]):
        props = skimage.measure.regionprops(mask_idx[fr,:,:,:])
        label, volume, X, Y, Z = [], [], [], [], []
        for prop in props:
            label.append(prop.label)
            volume.append(prop.area)
            z, y, x = prop.centroid
            X.append(x)
            Y.append(y)
            Z.append(z)           
        st = {'Label': label, "Channel": int(1), "Frame": int(fr +1), 'Volume': volume, 'X_pix': X, 'Y_pix': Y, 'Z_pix': Z}
        st = pd.DataFrame(st)
        summaryTable = summaryTable.append(st)
    return summaryTable 



def ilastik2mask(h5_path, invertCh):
    pathSansExt = os.path.splitext(h5_path)[0]
    ## func
    print("Open: " + h5_path)
    hf = h5py.File(h5_path, 'r')
    print(hf)
    hf.keys()
    img = hf.get('exported_data')
    img_ = img

    print(img.shape)
    mask = np.copy(img)

    # Invert backround to nuclear mask in ch1
    print("invert selected channels")
    for ch in range(img.shape[1]):
        for fr in range(img.shape[0]):
            if ch == (invertCh - 1):
                mask[fr,ch,:,:,:] = 1 - img[fr,ch,:,:,:] 
    
    # Make binary
    print("Make binary")
    mask[mask<0.5] = 0
    mask[mask>=0.5] = 1
    mask = mask.astype('bool')


    # Dialate and erode
    print("Dialate and erode")
    for ch in range(img.shape[1]):
        for fr in range(img.shape[0]):
            mask[fr,ch,:,:,:] = scipy.ndimage.morphology.binary_dilation(mask[fr,ch,:,:,:])
            mask[fr,ch,:,:,:] = scipy.ndimage.morphology.binary_erosion(mask[fr,ch,:,:,:])
    

    # Fill holes per 2D plane
    print("Fill holes in 2D")
    for ch in range(img.shape[1]):
        for fr in range(img.shape[0]):
            for sl in range(img.shape[2]):
                mask[fr,ch,sl,:,:] = scipy.ndimage.binary_fill_holes(mask[fr,ch,sl,:,:])

    # 3D connected components to make indexed image
    print("3D connected components")
    mask = mask.astype('uint8')
    for ch in range(img.shape[1]):
        for fr in range(img.shape[0]):
            mask[fr,ch,:,:,:] = cc3d.connected_components(mask[fr,ch,:,:,:]).astype('uint8')


    # Remove nucleus on edge
    print("Remove edge nucleus")
    for ch in range(img.shape[1]):
        for fr in range(img.shape[0]):
            img_tmp = mask[fr,ch,:,:,:]
            sel_mask = np.ones(img_tmp.shape, dtype=bool) # Make selection mask
            sel_mask[2:-2, 2:-2, 2:-2] = False # set middle to False
            edge_ids = img_tmp[sel_mask]
            edge_ids = edge_ids[edge_ids > 0]
            edge_ids = [i for i in set(edge_ids)]
            for idx in edge_ids:
                img_tmp[img_tmp == idx] = 0
            mask[fr,ch,:,:,:] = img_tmp

    # 3D connected components to make indexed image
    print("3D connected components again")
    mask = mask.astype('uint8')
    for ch in range(img.shape[1]):
        for fr in range(img.shape[0]):
            mask[fr,ch,:,:,:] = cc3d.connected_components(mask[fr,ch,:,:,:]).astype('uint8')
    
    #plt.imshow(mask[0,0,15,:,:], interpolation='nearest')
    #plt.show()

    # Make summary table
    print("Measure 3D")
    summary_table = makeSummaryTable(mask)
    summary_table.to_csv(pathSansExt + "_idx.csv", sep=',', index=False)    
    print("Save")
    #tiff.imsave(pathSansExt + "_idx.tif", mask.astype('uint8'), imagej=True)
    tiff.imsave(pathSansExt + "_idx.tif", np.swapaxes(mask,1,2).astype('uint8'), imagej=True)

#tczyx
#h5_path = r"M:\ChromatinTeam\Images_processing\20200602_OO_326_TAD-DNA_Fixed-FISHed_tmTit\DNA_stain\02_DC_4D_mask\IBIDI_2A_60deg_01_before_ABC_02_DNAOnly_DE_3_W0001_P0001_dc_mask.h5"
#invertCh = 1
#ilastik2mask(h5_path, invertCh)