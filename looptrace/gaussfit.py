#!/usr/bin/env python
"""
Routines for fitting gaussians

Adapted from Hazen Babcock/STORM-Analysis package.
"""

from typing import *

import numpy
import scipy
import scipy.optimize

from looptrace.numeric_types import *


def fitAFunctionLS(data, params, fn):
    """
    Least Squares fitting.
    """
    result = params
    errorfunction = lambda p: numpy.ravel(fn(*p)(*numpy.indices(data.shape)) - data)
    result = scipy.optimize.leastsq(errorfunction, params, full_output = 0, maxfev = 200)
    #err = errorfunction(result)
    #err = scipy.sum(err * err)
    #if (success < 1) or (success > 4):
        #print("Fitting problem!", success, mesg)
    #    good = False
    return result


def fitAFunctionMLE(data: numpy.ndarray, params: List[NumberLike], fn: callable) -> List[Union[numpy.ndarray, bool]]:
    """
    MLE fitting, following Laurence and Chromy.

    Parameters
    ----------
    data : numpy.ndarray
        The data to which the function is to be fit
    params : list
        The initial guess for the optimisation, an array-like
    fn : callable
        The optimisation function

    Returns
    -------
    A two-element list, in which the first element is a numpy array containing the result, and 
    the second element is a flag indicating whether the result should be considered 
    legitimate or not.
    """
    result = params
    def errorFunction(p):
        fit = fn(*p)(*numpy.indices(data.shape))
        t1 = 2.0 * numpy.sum(fit - data)
        t2 = 2.0 * numpy.sum(data * numpy.log(fit/data))
        return t1 - t2
    try:
        result, _, _, _, _, _ = scipy.optimize.fmin(errorFunction, params, full_output = 1, maxiter = 1000, disp = False)
    except:
        good = False
    else:
        good = True
    return [result, good]


#
# Fitting a gaussian, this is almost a verbatim copy of:
#
# http://www.scipy.org/Cookbook/FittingData#head-11870c917b101bb0d4b34900a0da1b7deb613bf7
#

def symmetricGaussian3D(bg, A, center_z, center_y, center_x, sigma_z, sigma_xy):
    return lambda z,y,x: bg + A*numpy.exp(-((x-center_x)**2/(2*sigma_xy)**2 +
                                            (y-center_y)**2/(2*sigma_xy)**2 +
                                            (z-center_z)**2/(2*sigma_z)**2))


def fitSymmetricGaussian3D(data, sigma, center=None):
    return _fit_3d_gaussian(fit_proc=fitAFunctionLS, img_data=data, sigma=sigma, center=center)


def fitSymmetricGaussian3DMLE(data: numpy.ndarray, sigma: NumberLike, center: Optional[Union[str, List[IntegerLike]]]):
    return _fit_3d_gaussian(fit_proc=fitAFunctionMLE, img_data=data, sigma=sigma, center=center)


def _fit_3d_gaussian(fit_proc: callable, img_data: numpy.ndarray, sigma: NumberLike, center: Optional[Union[str, List[IntegerLike]]]):
    """
    Data is assumed centered on the gaussian and of size roughly 2x the width.
    
    Parameters
    ----------
    fit_proc : callable
        The fitting procedure / optimisation strategy, e.g. maximum-likelihood estimation or least squares
    img_data : np.ndarray
        The array of pixel intensities to which to fit a Gaussian; this corresponds to a single round / 
        hybridisation / frame / probe, for a particular field of view and a particular channel. 
        Namely, then, it fixes FOV and 2 of the 5 dimensions, leaving (z,y,x), and hence this is a 3D 
        numpy array.
    sigma : NumberLike
        The standard deviation starting guess 
    center : Optional[Union[str, List[IntegerLike]]]
        The starting guess strategy for the distribution's center; can be a set of 3 coordinates (z, y, x), 
        or null to use the literal image center, or 'max' to say that the point of maximal pixel intensity 
        should be used. If real values, these should be integer-like since they represent pixel coordinates.
    
    Returns
    -------
    list
        A sequence of values corresponding to fitted values of the Gaussian function, optimising according 
        to the given fit_proc
    
    """
    if center is None:
        # Take the literal pixel center of the image.
        center = [s // 2 for s in img_data.shape]
    elif center == 'max':
        # Seed the fit procedure's center with the pixel coordinates of max signal intensity.
        center = list(numpy.unravel_index(numpy.argmax(img_data, axis=None), img_data.shape))
    init_guess = [numpy.min(img_data), numpy.max(img_data)] + center + [sigma, sigma]
    return fit_proc(img_data, init_guess, symmetricGaussian3D)
