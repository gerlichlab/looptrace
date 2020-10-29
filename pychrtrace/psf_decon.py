# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 06:20:30 2020

@author: ellenberg
"""

def psf_gen_GL():
    mp = msPSF.m_params
    mp['M']=63
    mp['NA']=1.46
    pixel_size = 0.1
    rv = np.arange(0.0, 3.01, pixel_size)
    zv = np.arange(-1.5, 1.51, pixel_size*2)
    psf_xyz = msPSF.gLXYZFocalScan(mp, pixel_size, 31, zv, pz = 0.0)
    psfSlicePics(psf_xyz, 15, 7, zv)
    return psf_xyz

def psfSlicePics(psf, sxy, sz, zvals, pixel_size = 0.1):
    ex = pixel_size * 0.5 * psf.shape[1]
    fig = plt.figure(figsize = (12,4))
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(np.sqrt(psf[sz,:,:]),
               interpolation = 'none', 
               extent = [-ex, ex, -ex, ex],
               cmap = "gray")
    ax1.set_title("PSF XY slice")
    ax1.set_xlabel(r'x, $\mu m$')
    ax1.set_ylabel(r'y, $\mu m$')

    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(np.sqrt(psf[:,:,sxy]),
               interpolation = 'none',
               extent = [-ex, ex, zvals.max(), zvals.min()],
               cmap = "gray")
    ax2.set_title("PSF YZ slice")
    ax2.set_xlabel(r'y, $\mu m$')
    ax2.set_ylabel(r'z, $\mu m$')

    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow(np.sqrt(psf[:,sxy,:]), 
               interpolation = 'none',
               extent = [-ex, ex, zvals.max(), zvals.min()],
               cmap = "gray")
    ax3.set_title("PSF XZ slice")
    ax3.set_xlabel(r'x, $\mu m$')
    ax3.set_ylabel(r'z, $\mu m$')

    plt.show()
    
def RL_decon(data, kernel, iterations):
    algo = fd_restoration.RichardsonLucyDeconvolver(data.ndim).initialize()
    res = algo.run(fd_data.Acquisition(data=data, kernel=kernel), niter=iterations).data
    return res