#!/usr/bin/env python
"""
Routines for fitting gaussians

Adapted from Hazen Babcock/STORM-Analysis
"""

import numpy
import scipy
import scipy.optimize

def fitAFunctionLS(data, params, fn):
    """
    Least Squares fitting.
    """
    result = params
    errorfunction = lambda p: numpy.ravel(fn(*p)(*numpy.indices(data.shape)) - data)
    good = True
    [result, cov_x, infodict, mesg, success] = scipy.optimize.leastsq(errorfunction, params, full_output = 1, maxfev = 500)
    err = errorfunction(result)
    err = scipy.sum(err * err)
    if (success < 1) or (success > 4):
        #print("Fitting problem!", success, mesg)
        good = False
    return [result, good]

def fitAFunctionMLE(data, params, fn):
    """
    MLE fitting, following Laurence and Chromy.
    """
    result = params
    def errorFunction(p):
        fit = fn(*p)(*numpy.indices(data.shape))
        t1 = 2.0 * numpy.sum(fit - data)
        t2 = 2.0 * numpy.sum(data * numpy.log(fit/data))
        return t1 - t2
    good = True
    try:
        [result, fopt, iter, funcalls, warnflag] = scipy.optimize.fmin(errorFunction, params, full_output = 1, maxiter = 1000, disp = False)
    except:
        warnflag = 1
    if (warnflag != 0):
        #print("Fitting problem!")
        good = False
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
    """
    Data is assumed centered on the gaussian and of size roughly 2x the width.
    """
    params = [numpy.min(data),
              numpy.max(data)]
    if center is None:
        center = [s//2 for s in data.shape]
    elif center == 'max':
        center = list(numpy.unravel_index(numpy.argmax(data, axis=None), data.shape))
    params += center
    params += [sigma,
               sigma]
    return fitAFunctionLS(data, params, symmetricGaussian3D)

def fitSymmetricGaussian3DMLE(data, sigma, center):
    """
    Data is assumed centered on the gaussian and of size roughly 2x the width.
    """
    params = [numpy.min(data),
              numpy.max(data)]
    params += center
    params += [sigma,
              sigma]
    return fitAFunctionMLE(data, params, symmetricGaussian3D)